import warnings
warnings.filterwarnings("ignore")

import os
import h5py
import re
import collections
import torch
import requests
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import MDAnalysis as mda

from functools import partial
from tqdm import tqdm
from typing import Union, Literal, Any, Optional

from MDAnalysis.lib.distances import calc_dihedrals
from graphein.protein.config import ProteinGraphConfig, DSSPConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import (
    add_aromatic_interactions, 
    add_disulfide_interactions, 
    add_hydrogen_bond_interactions, 
    add_peptide_bonds, 
    add_hydrophobic_interactions,
    add_ionic_interactions,
    add_k_nn_edges
)
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, expasy_protein_scale
from graphein.protein.features.nodes import asa, rsa
from graphein.protein.features.nodes.dssp import secondary_structure
from graphein.protein.resi_atoms import RESI_THREE_TO_1

from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dropout_edge, to_undirected
from torchmetrics.functional.pairwise import pairwise_euclidean_distance

from enum import Enum

__all__ = ["AntigenDataset", "ProteinEdgeTypeFuncs", "ProteinNodeMetaDataFuncs", "ProteinGraphMetaDataFuncs"]

class ProteinEdgeTypeFuncs(Enum):
    AROMATIC_INTERACTIONS = add_aromatic_interactions
    DISULFIDE_INTERACTIONS = add_disulfide_interactions
    HYDROGEN_BOND_INTERACTIONS = add_hydrogen_bond_interactions
    PEPTIDE_BONDS = add_peptide_bonds
    HYDROPHOBIC_INTERACTIONS = add_hydrophobic_interactions
    IONIC_INTERACTIONS = add_ionic_interactions
    K_NN_EDGES = add_k_nn_edges

    @classmethod
    def get_members(cls):
        return [
            cls.AROMATIC_INTERACTIONS, 
            cls.DISULFIDE_INTERACTIONS,
            cls.HYDROGEN_BOND_INTERACTIONS,
            cls.PEPTIDE_BONDS,
            cls.HYDROPHOBIC_INTERACTIONS,
            cls.IONIC_INTERACTIONS,
            cls.K_NN_EDGES
        ]

class ProteinNodeMetaDataFuncs(Enum):
    AMINO_ACID_ONE_HOT = amino_acid_one_hot
    ISOELECTRIC_POINTS = partial(expasy_protein_scale, add_separate=True, selection=["isoelectric_points"])

    @classmethod
    def get_members(cls):
        return [
            cls.AMINO_ACID_ONE_HOT,
            cls.ISOELECTRIC_POINTS.value
        ]

class ProteinGraphMetaDataFuncs(Enum):
    ASA = asa
    RSA = rsa
    SECONDARY_STRUCTURE = secondary_structure

    @classmethod
    def get_members(cls):
        return [
            cls.ASA,
            cls.RSA,
            cls.SECONDARY_STRUCTURE
        ]

class AntigenDataset(Dataset):
    def __init__(self, 
                root: str,
                antigens_csv_path: os.PathLike,
                esm2_embedding_path: os.PathLike,
                prott5_embedding_path: os.PathLike,
                esm3_embedding_path: os.PathLike,
                amino_acid_physicochemical_features_path: os.PathLike,
                transform = Optional[None], 
                pre_transform = Optional[None], 
                edge_types: Optional[Union[list[ProteinEdgeTypeFuncs], None]] = None,
                node_metadata_funcs: Optional[Union[list[ProteinNodeMetaDataFuncs], None]] = None,
                graph_metada_funcs: Optional[Union[list[ProteinGraphMetaDataFuncs], None]] = None,
                k: Optional[int] = 10,
                augment_data: Optional[bool] = True,
                node_attributes: Optional[Union[list[str], Literal["raw", "raw+dihedrals", "raw+dihedrals+embeddings"], None]] = None,
                edge_attributes: Optional[Union[list[str], None]] = None,
                recreate_processed_dir: Optional[bool] = False,
                embedding_model: Optional[Literal["esm2", "prott5", "esm3"]] = "esm3",
                is_alphafold2_predictions: Optional[bool] = False,
                **kwargs):
        
        """
        Initializes the GraphBepiDataset.
        Args:
            root (str): Root directory where the dataset should be stored.
            antigens_csv_path (os.PathLike): Path to the CSV file containing antigen's pdb code, chain and epitope annotations.
            esm2_embedding_path (os.PathLike): Path to the ESM2 embedding file. Should be in .h5 format.
            prott5_embedding_path (os.PathLike): Path to the ProtT5 embedding file. Should be in .h5 format.
            esm3_embedding_path (os.PathLike): Path to the ESM3 embedding file. Should be in .h5 format.
            amino_acid_physicochemical_features_path (os.PathLike): Path to the file containing amino acid physicochemical features.
            transform (Optional[None]): A function/transform that takes in an object and returns a transformed version. Default is None.
            pre_transform (Optional[None]): A function/transform that takes in an object and returns a transformed version before any other processing. Default is None.
            edge_types (Optional[Union[list[ProteinEdgeTypeFuncs], None]]): List of functions from `graphein` to define edge types. 
            If it is None, uses all methods in the `ProteinEdgeTypeFuncs`. Default is None.
            node_metadata_funcs (Optional[Union[list[ProteinNodeMetaDataFuncs], None]]): List of functions from `graphein` to define node metadata. 
            If it is None, uses all methods in the `ProteinNodeMetaDataFuncs`. Default is None.
            graph_metada_funcs (Optional[Union[list[ProteinGraphMetaDataFuncs], None]]): List of functions to from `graphein` define graph metadata. 
            If it is None, uses all methods in the `ProteinGraphMetaDataFuncs`. Default is None.
            k (Optional[int]): K parameter for KNN edge type. Default is 10.
            augment_data (Optional[bool]): Whether to augment antigen graph. If it is True, dataset will return list of `torch_geometric.data.Data` object.
            Augmentations are calculated on-the-fly using `get` function. Defaut augmentations are `node_dropping` amd `crop_graph`. Default is True.
            node_attributes (Optional[Union[list[str], Literal["raw", "raw+dihedrals", "raw+dihedrals+embeddings"], None]]): List of node attributes that will be included `torch_geometric.data.Data` object. 
            If it is None, selects all node attrites (check `get_node_attributes` function). Default is None.
            edge_attributes (Optional[Union[list[str], None]]): List of edge attributes that will be included `torch_geometric.data.Data` object. 
            If it is None, selects all edge attrites (check `get_edge_attributes` function). Default is None.
            recreate_processed_dir (Optional[bool]): Whether to recreate the processed directory in dataset folder. 
            If user want to compare model results with different attribute combinations without creating dataset each time, this parameter can be set True.
            Recreates all the `torch_geometric.data.Data` objects in processed directory with new set of given parameters. Default is False.
            embedding_model (Optional[Literal["esm2", "prott5", "esm3"]]): Embedding model include as node attribute. Default is "esm3".
            is_alphafold2_predictions (Optional[bool]): Whether the dataset will be constructed from Alphafold2 predictions. Default is False.
            **kwargs: Additional keyword arguments for `torch_geometric.data.Dataset`.
        Raises:
            ValueError: If an invalid embedding model is provided.
        """
        
        self.augment_data = augment_data

        self.antigens_csv_path = antigens_csv_path
        self.esm2_embedding_path = esm2_embedding_path
        self.esm3_embedding_path = esm3_embedding_path
        self.prott5_embedding_path = prott5_embedding_path
        self.embedding_model = embedding_model
        self.amino_acid_physicochemical_features_path = amino_acid_physicochemical_features_path
        self.is_alphafold2_predictions = is_alphafold2_predictions
        
        self.set_edge_types(edge_types, k)

        self.set_node_metadata_funcs(node_metadata_funcs)
        self.set_graph_metada_funcs(graph_metada_funcs)
        
        if embedding_model in ["esm2", "prott5", "esm3"]:
            # get esm2 embeddings
            self.get_esm2_embeddings()
            # get prott5 embeddings
            self.get_prott5_embeddings()
            # get esm3 embeddings
            self.get_esm3_embeddings()
        else:
            raise ValueError("Invalid embedding model. Please choose 'esm2', 'esm3' or 'prott5'.")

        # define node and edge attributes
        self.get_node_attributes(node_attributes)
        self.get_edge_attributes(edge_attributes)
            

        # define embedding model to shape dictionary
        self.embedding_model_to_shape = {
            "prott5":1024,
            "esm2":1280,
            "esm3":1536,
        }
        
        super().__init__(root, transform, pre_transform, **kwargs)

        if recreate_processed_dir:
            self.recreate_processed_dir()


    @property
    def preprocessed_dir(self) -> str:
        dir_path = os.path.join(self.root, "preprocessed")
        if os.path.isdir(dir_path):
            return dir_path
        else:
            os.makedirs(dir_path)
            return dir_path
        
    @property
    def antigens_df(self) -> pd.DataFrame:
        antigens_df = pd.read_csv(self.antigens_csv_path, sep=",")
        return antigens_df
    
    @property
    def amino_acid_physicochemical_features(self) -> dict[str, np.array]:
        df = pd.read_csv(self.amino_acid_physicochemical_features_path)

        pattern = r"\b(\w+(?:\s+\w+)*)|\w+\s*\(\d+\)"  # Matches word+ (optional whitespace+word+)* OR word+ (optional whitespace+number in parentheses)

        for column_name in df.columns[3:]:
            df[column_name] = df[column_name].apply(lambda x: re.search(pattern, x).group(1))

        df["Abbreviation 3 letter"] = df["Abbreviation 3 letter"].apply(lambda x: x.upper())

        amino_acid_physicochemical_features = collections.defaultdict(dict)
        for column_name in df.columns[3:]:
            class_labels = df[column_name].unique().tolist()
            for aa, feature in df[["Abbreviation 3 letter", column_name]].values:
                amino_acid_physicochemical_features[re.search(pattern, column_name).group(1)][aa] = self.one_hot_encode([feature], class_labels)[0]

        return amino_acid_physicochemical_features

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{pdb.upper()}.pdb" for pdb in self.antigens_df["PDB"].to_list()]    

    @property
    def processed_file_names(self) -> list[str]:
        return [f"data_{index}.pt" for index, _ in enumerate(self.raw_file_names)]


    def len(self) -> int:
        return len(self.raw_file_names)
    
    @property
    def class_distribution_stats(self) -> dict[int, int]:
        """
        Computes the distribution of classes in the dataset.
        This property iterates through all processed files, loads the data,
        and counts the occurrences of each class label (0 and 1).
        Returns:
            dict[int, int]: A dictionary with the class labels as keys and 
                            their respective counts as values.
                            
        """
        zeros = 0
        ones =  0
        for file in self.processed_file_names:
            data = torch.load(os.path.join(self.processed_dir, file))
            zeros += (data.y == 0.).sum()
            ones += data.y.sum()
        
        return {0:zeros.int().item(), 1:ones.int().item()}
    
    def download(self) -> None:
        """
        Downloads PDB files for antigens listed in the antigens DataFrame.

        If the attribute `is_alphafold2_predictions` is True, the method does nothing.
        Otherwise, it iterates over unique PDB identifiers in the `antigens_df` DataFrame,
        downloads the corresponding PDB files from the RCSB PDB repository, and saves them
        to the specified raw directory.
        """
        if self.is_alphafold2_predictions:
            pass
        else:
            for pdb in tqdm(set(self.antigens_df["PDB"].to_list()), desc="Downloading pdb files"):
                file_name = f"{pdb.upper()}.pdb"
                content = requests.get(f"https://files.rcsb.org/download/{file_name}").content

                output_full_path = os.path.join(self.raw_dir, f"{file_name}")
                with open(output_full_path, "wb") as file:
                    file.write(content)



    def process(self) -> None:
        """
        Processes antigen graphs by converting them to NetworkX graphs, saving them as pickle files,
        creating PyTorch Geometric data objects, and saving these objects as .pt files.
        This method performs the following steps:
        1. Iterates over antigen graphs constructed using NetworkX.
        2. Saves each NetworkX graph as a pickle file in the preprocessed directory.
        3. Converts each NetworkX graph to a PyTorch Geometric data object.
        4. Saves each PyTorch Geometric data object as a .pt file in the processed directory.
        """
        
        for index, antigen_graph in tqdm(enumerate(self.construct_networkx_graphs()), desc="Processing networkx graphs"):
            
            with open(os.path.join(self.preprocessed_dir, f"graph_{index}.pickle"), "wb") as file:
                pickle.dump(antigen_graph, file)

            data = self.create_pytorch_geometric_data_object(g=antigen_graph)
                
            torch.save(data, os.path.join(self.processed_dir, f"data_{index}.pt"))
        
    
    def recreate_processed_dir(self) -> None:
        """
        This function recreates the processed directory by removing existing processed files and recreating them.
        It iterates through the preprocessed directory, loads each pickle file, creates a PyTorch Geometric data object,
        and saves the data object as a .pt file in the processed directory.

        """
        if not os.path.isdir(self.processed_dir):
            raise FileNotFoundError("Processed directory does not exist. Do not use `recreate_processed_dir` if you have not created the dataset directories yet.")
        
        if not os.path.isdir(self.preprocessed_dir):
            raise FileNotFoundError("Preprocessed directory does not exist. Do not use `recreate_processed_dir` if you have not created the dataset directories yet.")
        
        if not os.listdir(self.processed_dir):
            raise FileNotFoundError("Processed directory is empty. Do not use `recreate_processed_dir` if you have not processed the dataset yet.")
        
        if not os.listdir(self.preprocessed_dir):
            raise FileNotFoundError("Preprocessed directory is empty. Do not use `recreate_processed_dir` if you have not processed the dataset yet.")
        

        print("Recreating processed directory")

        for processed_file in self.processed_paths:
            os.remove(processed_file)

        for index in tqdm(range(len(os.listdir(self.preprocessed_dir))), desc="Recreating processed directory"):

            with open(os.path.join(self.preprocessed_dir, f"graph_{index}.pickle"), "rb") as f:
                print(f"graph_{index}.pickle")
                g = pickle.load(f)

                change_embedding = False
                for _, d in g.nodes(data=True):
                    if d["embedding"].shape[0] != self.embedding_model_to_shape[self.embedding_model]:
                        change_embedding = True
                    
                    break
                
                if change_embedding:
                    for _, d in g.nodes(data=True):
                        d["embedding"] = d[f"{self.embedding_model}_embedding"]
                        

                data = self.create_pytorch_geometric_data_object(g)
                torch.save(data, os.path.join(self.processed_dir, f"data_{index}.pt"))


    def get(self, index) -> Union[list[Data], Data]:
        """
        Retrieve data and optionally augment it.
        Parameters:
            index (int): The index of the data to retrieve.
        Returns:
            Union[list[Data], Data]: If augment_data is True, returns a list containing the original data and two augmented views.
                        Otherwise, returns the original data.
        """
        if self.augment_data:

            data = torch.load(os.path.join(self.processed_dir, f"data_{index}.pt"))

            with open(os.path.join(self.preprocessed_dir, f"graph_{index}.pickle"), "rb") as f:
                view1 = pickle.load(f)

            view1 = self.create_pytorch_geometric_data_object(self.node_dropping(view1, p=0.25))
            
            with open(os.path.join(self.preprocessed_dir, f"graph_{index}.pickle"), "rb") as f:
                view2 = pickle.load(f)

            view2 = self.create_pytorch_geometric_data_object(self.crop_graph(view2, crop_ratio=0.75, crop_start_part="epitope_centric"))

            return [data, view1, view2] 
                

        else:
            return torch.load(os.path.join(self.processed_dir, f"data_{index}.pt"))
        

    # def node_dropping(self, data: Data, p: float = 0.5, remove_epitope_nodes: bool = False) -> Data:
    #     """
    #     This function performs node dropping on a torch geometric data object.

    #     Parameters:
    #     - data (torch_geometric.data.Data): The graph data object to perform node dropping on.
    #     - p (float, optional): The probability of dropping a node. Default is 0.5.
    #     - remove_epitope_nodes (bool, optional): Whether to remove epitope nodes in graph. Default is False.

    #     Returns:
    #     - Data: The graph data object after node dropping.

    #     Note:
    #     - The function updates the x, y, and pos attributes of the data object based on the selected nodes.
    #     - The function updates the edge_index attribute of the data object to reflect the selected edges.
    #     - If the data object has edge_attr attribute, the function updates it based on the selected edges.
    #     """
    #     data = data.clone()

    #     p = round(1 - p, 1)
    #     selection =  torch.from_numpy(np.random.binomial(1, p, data.x.size(0)))

    #     if remove_epitope_nodes is False:
    #         selection[(data.y == 1) & (selection == 0)] = 1

    #     assert data.x.shape[0] == selection.shape[0]

    #     data.x = data.x[selection.bool()]
    #     data.y = data.y[selection.bool()]

    #     if hasattr(data, "pos"):
    #         data.pos = data.pos[selection.bool()]

    #     assert data.x.shape[0] == data.y.shape[0], "x and y shape mismatch in Data object"

    #     retained_indices = torch.nonzero(selection, as_tuple=False).squeeze()
    #     assert selection.sum().item() == retained_indices.shape[0]

    #     mask = torch.isin(data.edge_index[0], retained_indices) & torch.isin(data.edge_index[1], retained_indices)

    #     edge_index_selected = data.edge_index[:, mask]

    #     index_mapping = {retained_index.item(): new_index for new_index, retained_index in enumerate(retained_indices)}

    #     for old_index, new_index in index_mapping.items():
    #         edge_index_selected[edge_index_selected == old_index] = new_index
    #         # edge_index_selected = torch.where((edge_index_selected == old_index), torch.tensor(new_index), edge_index_selected)

    #     edge_index_selected = edge_index_selected.to(torch.long)
    #     data.edge_index = edge_index_selected

    #     if hasattr(data, "edge_attr"):
    #         data.edge_attr = data.edge_attr[mask]
    #         assert data.edge_index.shape[1] == data.edge_attr.shape[0], "edge_index shape and edge_attr shape mismatch in Data object"

    #     return data
    
    def node_dropping(self, 
                      g: nx.Graph, 
                      p: float = 0.5, 
                      shuffle: bool = False, 
                      remove_epitope_nodes: bool = False, 
                      return_coords: bool = False) -> Union[tuple[nx.Graph, np.array] ,nx.Graph]:
        """
        This function performs node dropping on a given NetworkX graph.
        
        Parameters:
            g (nx.Graph): The input graph on which node dropping will be performed.
            p (float, optional): The probability of dropping a node using samples from a Bernoulli distribution. Default is 0.5.
            shuffle (bool, optional): If True, the order of nodes will be shuffled before dropping. Default is False.
        
        Returns:
            nx.Graph: The modified graph after node dropping.    
        """
        g = g.copy()
        p = round(1 - p, 2)
        probs =  np.random.binomial(1, p, len(g.nodes))
        nodes_arr = np.array(list(g.nodes))
        
        if shuffle:
            np.random.shuffle(nodes_arr)

        drop_nodes = nodes_arr[(probs == 0.)].tolist()

        if remove_epitope_nodes is False:
            drop_nodes = [n for n, d in g.nodes(data=True) if n in drop_nodes and d["epitope"] != 1]

        g.remove_nodes_from(drop_nodes)

        if list(nx.isolates(g)):
            g.remove_nodes_from(list(nx.isolates(g)))

        if return_coords:
            coords = np.array([d["coords"] for _, d in g.nodes(data=True)])
            return g, coords
        else:
            return g

    # def edge_dropping(self, data: Data, p: float = 0.6, to_undirected_graph: bool = False) -> Data:
    #     """
    #     Performs edge dropping on a torch geometric data object.

    #     Parameters:
    #     - data (torch_geometric.data.Data): The graph data object to perform edge dropping on.
    #     - p (float, optional): The probability of dropping an edge. Default is 0.6.
    #     - to_undirected_graph (bool, optional): Whether to convert the graph to an undirected graph. Default is False.

    #     Returns:
    #     - Data: The graph data object after edge dropping.

    #     Note:
    #     - The function updates the edge_index attribute of the data object to reflect the selected edges.
    #     - If the data object has edge_attr attribute, the function updates it based on the selected edges.
    #     - If the data object has isolated nodes after edge dropping, the function removes them.
    #     """

    #     data = data.clone()
    #     edge_index, edge_mask = dropout_edge(data.edge_index, p, force_undirected=False)
    #     data.edge_index = edge_index

    #     if hasattr(data, "edge_attr"):
    #         data.edge_attr = data.edge_attr[edge_mask]

    #     if data.has_isolated_nodes():
    #         t = RemoveIsolatedNodes()
    #         data = t(data)

    #     assert not data.has_isolated_nodes(), "Data object has isolated nodes"

    #     if to_undirected_graph:
    #         data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)
    #         assert not data.is_directed(), "Data object is directed"

    #     return data
        
    # def edge_dropping(self, g: nx.Graph, p: float = 0.5, shuffle: bool = False, return_coords: bool = False) -> Union[tuple[nx.Graph, np.array] ,nx.Graph]:
    #     """
    #     This function performs edge dropping on a given NetworkX graph.
        
    #     Parameters:
    #         g (nx.Graph): The input graph on which node dropping will be performed.
    #         p (float, optional): The probability of dropping a edge using samples from a Bernoulli distribution. Default is 0.5.
    #         shuffle (bool, optional): If True, the order of nodes will be shuffled before dropping. Default is False.
        
    #     Returns:
    #         nx.Graph: The modified graph after node dropping.    
    #     """
    #     g = g.copy()
    #     p = round(1 - p, 1)
    #     probs =  np.random.binomial(1, p, len(g.edges))
    #     edges_arr = np.array(list(g.edges))

    #     if shuffle:
    #         np.random.shuffle(edges_arr)
        
    #     drop_edges = edges_arr[(probs == 0.)].tolist()

    #     g.remove_edges_from(drop_edges)

    #     if list(nx.isolates(g)):
    #         g.remove_nodes_from(list(nx.isolates(g)))

    #     if return_coords:
    #         coords = np.array([d["coords"] for _, d in g.nodes(data=True)])
    #         return g, coords
    #     else:
    #         return g

    # def crop_graph(self, data: Data, crop_ratio: float = 0.5, crop_start_part: Literal["bottom", "top", "center", "epitope_centric"] = "top") -> Data:
    #     """
    #     Crops a protein graph based on the specified crop ratio and crop start part.

    #     Args:
    #     - data (torch_geometric.data.Data): The graph data object to perform cropping.
    #     - crop_ratio (float, optional): The ratio of the maximum distance of the graph to be cropped. Defaults to 0.5.
    #     - crop_start_part (Literal["bottom", "top", "center", "epitope_centric"], optional): The part of the graph from which the cropping will start. Defaults to "top".

    #     Returns:
    #     - Data: The graph data object after cropping.

    #     Raises:
    #     - ValueError: If the crop ratio is less than or equal to zero or greater than 1.
    #     """

    #     if crop_ratio <= 0 or crop_ratio > 1:
    #         raise ValueError("Crop ratio must be greater than zero and less than 1")

    #     data = data.clone()

    #     euc_dist_matrix = pairwise_euclidean_distance(data.pos)
    #     max_distance_of_graph = euc_dist_matrix.max().item()

    #     # Calculate the crop distance based on the maximum distance and the crop ratio
    #     crop_distance = max_distance_of_graph * crop_ratio

    #     max_dist_loc = (euc_dist_matrix==torch.max(euc_dist_matrix)).nonzero()
    #     max_row_idx = max_dist_loc[0][0].item()
    #     max_col_idx = max_dist_loc[0][1].item()

    #     if torch.gt(data.pos[max_row_idx], data.pos[max_col_idx]).sum().item() > torch.gt(data.pos[max_col_idx], data.pos[max_row_idx]).sum().item():
    #     # if data.pos[max_row_idx][1].item() > data.pos[max_col_idx][1].item():
    #         top_loc = data.pos[max_row_idx]
    #         bottom_loc = data.pos[max_col_idx]
    #     else:
    #         top_loc = data.pos[max_col_idx]
    #         bottom_loc = data.pos[max_row_idx]

    #     # Determine the selected location based on the crop start part
    #     if crop_start_part == "bottom":
    #         selected_loc = bottom_loc
    #     elif crop_start_part == "top":
    #         selected_loc = top_loc
    #     elif crop_start_part == "center":
    #         # Calculate the center location
    #         crop_distance /= 1.5
    #         selected_loc = torch.mean(data.pos, dim=0)
    #     else:
    #         epitope_pos = torch.index_select(data.pos, 0, (data.y == 1).nonzero().squeeze())
    #         selected_loc = torch.mean(epitope_pos, dim=0)
    #         crop_distance /= 1.5


    #     # Create a list of node indexes that are within the crop location
    #     selection = []
    #     for loc in data.pos:
    #         euclidean_distance = torch.sqrt(torch.sum(torch.square(selected_loc - loc))).item()
    #         if torch.equal(selected_loc, loc):
    #             selection.append(1)
    #         elif euclidean_distance <= crop_distance:
    #             selection.append(1)
    #         else:
    #             selection.append(0)
        
    #     selection = torch.tensor(selection)

    #     assert data.x.shape[0] == selection.shape[0]

    #     data.x = data.x[selection.bool()]
    #     data.y = data.y[selection.bool()]

    #     if hasattr(data, "pos"):
    #         data.pos = data.pos[selection.bool()]

    #     assert data.x.shape[0] == data.y.shape[0], "x and y shape mismatch in Data object"

    #     retained_indices = torch.nonzero(selection, as_tuple=False).squeeze()
    #     assert selection.sum().item() == retained_indices.shape[0]

    #     mask = torch.isin(data.edge_index[0], retained_indices) & torch.isin(data.edge_index[1], retained_indices)

    #     edge_index_selected = data.edge_index[:, mask]

    #     index_mapping = {retained_index.item(): new_index for new_index, retained_index in enumerate(retained_indices)}

    #     for old_index, new_index in index_mapping.items():
    #         edge_index_selected[edge_index_selected == old_index] = new_index
    #         # edge_index_selected = torch.where((edge_index_selected == old_index), torch.tensor(new_index), edge_index_selected)
        
    #     edge_index_selected = edge_index_selected.to(torch.long)
    #     data.edge_index = edge_index_selected

    #     if hasattr(data, "edge_attr"):
    #         data.edge_attr = data.edge_attr[mask]
    #         assert data.edge_index.shape[1] == data.edge_attr.shape[0], "edge_index shape and edge_attr shape mismatch in Data object"

    #     return data
        
    def crop_graph(self, 
                   g: nx.Graph, 
                   crop_ratio: float = 0.6, 
                   crop_start_part: Literal["bottom", "top", "center", "epitope_centric"] = "epitope_centric", 
                   return_coords: bool = False) -> Union[tuple[nx.Graph, np.array], nx.Graph]:
        """
        This function crops a given NetworkX graph based on the specified crop ratio and crop start part. First, it takes maximum euclidean 
        distance between amino acids of given protein graph. Then, depending on crop ratio and start part, it crops the graph.


        Args:
            g (nx.Graph): The input graph to be cropped.
            crop_ratio (float, optional): The ratio of the maximum distance of the graph to be cropped. Defaults to 0.6.
            crop_start_part (Literal["bottom", "top", "center", "epitope_centric"], optional): The part of the graph from which the cropping will start. Defaults to "bottom".
            return_coords: (bool, optional): Whether to return the amino acid coordinates of the cropped graph.
        Returns:
            nx.Graph: The cropped graph.

        Raises:
            ValueError: If the input graph is empty or if the crop ratio is less than or equal to zero or greater than 1.

        """

        if not g:
            raise ValueError("Input graph is empty.")

        if crop_ratio <= 0 or crop_ratio > 1:
            raise ValueError("Crop ratio must be greater than zero and less than 1")
                            
        g = g.copy()

        # Calculate the maximum distance of the graph
        max_distance_of_graph = g.graph["dist_mat"].max().max()

        # Find the row and column indices of the maximum distance element
        max_row_idx, max_col_idx = np.unravel_index(
            np.argmax(g.graph["dist_mat"].to_numpy(), axis=None), g.graph["dist_mat"].to_numpy().shape
        )

        assert max_distance_of_graph == g.graph["dist_mat"].iloc[max_row_idx, max_col_idx], "Maximum distance of the graph is not equal to determined row and column indices of the maximum distance element"

        # Calculate the crop distance based on the maximum distance and the crop ratio
        crop_distance = max_distance_of_graph * crop_ratio
        
        if g.graph["coords"][max_row_idx][1] > g.graph["coords"][max_col_idx][1]:
            top_loc = g.graph["coords"][max_row_idx]
            bottom_loc = g.graph["coords"][max_col_idx]
        else:
            top_loc = g.graph["coords"][max_col_idx]
            bottom_loc = g.graph["coords"][max_row_idx]
        
        # Determine the selected location based on the crop start part
        if crop_start_part == "bottom":
            selected_loc = bottom_loc
        elif crop_start_part == "top":
            selected_loc = top_loc
        elif crop_start_part == "center":
            # Calculate the center location
            selected_loc = np.mean(g.graph["coords"], axis=0)
            crop_distance /= 1.5
        else:
            epitope_pos = np.array([d["coords"] for _, d in g.nodes(data=True) if d["epitope"] == 1])
            selected_loc = np.mean(epitope_pos, axis=0)
            crop_distance /= 1.5

        # Create a list of node indexes that are within the crop location
        selected_node_indexes = []
        coords = []
        for index, loc in enumerate(g.graph["coords"]):
            # Calculate the Euclidean distance between the selected location and each node location
            euclidean_distance = np.sqrt(np.sum(np.square(selected_loc - loc)))
            if np.array_equal(selected_loc, loc):
                selected_node_indexes.append(index)
                coords.append(loc)
            elif euclidean_distance <= crop_distance:
                selected_node_indexes.append(index)
                coords.append(loc)

        self.selected_indices_nx = selected_node_indexes
        # Create a subgraph of the nodes in the selected node indexes
        g_sub = g.subgraph([n for idx, n in enumerate(g.nodes) if idx in selected_node_indexes])

        # remove freeze
        g_sub = g_sub.copy()

        if list(nx.isolates(g_sub)):
            g_sub.remove_nodes_from(list(nx.isolates(g_sub)))

        if return_coords:
            coords = np.array([d["coords"] for _, d in g.nodes(data=True)])
            return g_sub, np.array(coords)
        else:
            return g_sub

    def set_edge_types(self, edge_types: Union[list[ProteinEdgeTypeFuncs], None], k: int) -> None:
        if edge_types:
            self.edge_types = [partial(ProteinEdgeTypeFuncs.K_NN_EDGES, k=k) if edge_type == ProteinEdgeTypeFuncs.K_NN_EDGES else edge_type for edge_type in edge_types]
        else:
            self.edge_types = [partial(ProteinEdgeTypeFuncs.K_NN_EDGES, k=k) if edge_type == ProteinEdgeTypeFuncs.K_NN_EDGES else edge_type for edge_type in ProteinEdgeTypeFuncs.get_members()]


    def set_node_metadata_funcs(self, node_metadata_funcs: Union[list[ProteinNodeMetaDataFuncs], None]) -> None:
        if node_metadata_funcs:
            self.node_metadata_funcs = [metadata.value if metadata == ProteinNodeMetaDataFuncs.ISOELECTRIC_POINTS else metadata for metadata in node_metadata_funcs]
        else:
            self.node_metadata_funcs = [metadata for metadata in ProteinNodeMetaDataFuncs.get_members()]

    def set_graph_metada_funcs(self, graph_metada_funcs: Union[list[ProteinGraphMetaDataFuncs], None]) -> None:
        if graph_metada_funcs:
            self.graph_metada_funcs = graph_metada_funcs
        else:
            self.graph_metada_funcs = [metadata for metadata in ProteinGraphMetaDataFuncs.get_members()]

    def construct_networkx_graphs(self) -> list[nx.Graph]:
        
        antigen_networkx_graphs = []
        for pdb, chain in tqdm(self.antigens_df[["PDB", "CHAIN"]].values, desc="Creating networkx graphs from pdb files"):
            path = os.path.join(self.raw_dir, f"{pdb.upper()}_{chain}.pdb" if self.is_alphafold2_predictions else f"{pdb.upper()}.pdb")

            g = self.construct_network_graph(path=path, pdb_code=pdb, chain=chain)
            
            antigen_networkx_graphs.append(g)
        
        return antigen_networkx_graphs

    def construct_network_graph(self, path: str, pdb_code: str, chain: str) -> nx.Graph:
        edge_construction_funcs = {"edge_construction_functions": self.edge_types}
        
        node_metadata_funcs = {"node_metadata_functions": self.node_metadata_funcs}
        graph_metada_funcs = {"graph_metadata_functions": self.graph_metada_funcs}
        dssp_config = {"dssp_config":DSSPConfig(executable="dssp")}

        config = ProteinGraphConfig(**edge_construction_funcs, **node_metadata_funcs, **graph_metada_funcs,
                                    **dssp_config, **{"verbose":False})
        
        if self.is_alphafold2_predictions:
            g = construct_graph(config=config, path=path, chain_selection=["A"])
        else:
            g = construct_graph(config=config, path=path, chain_selection=[chain])

        g = self.process_networkx_graph(g=g, 
                                    epitopes_dict=self.create_epitopes_dict_of_antigen(antigens_df=self.antigens_df, pdb_code=pdb_code, chain=chain),
                                    chain_id="A" if self.is_alphafold2_predictions else chain,
                                    pdb_path=path)

        return g

    def process_networkx_graph(self, 
                               g: nx.Graph, 
                               epitopes_dict: dict[str, str],
                               chain_id: str,
                               pdb_path: str, 
                               unique_ss: Union[list[str] | None] = None,
                               unique_edge_types: list[str] = None,
                               ) -> nx.Graph:
        """
        Processes NetworkX graphs
        
        Args:
            g (nx.Graph): A NetworkX graph to be scaled.
            unique_ss (list, optional): A list of unique secondary structure labels. Defaults to None.
            epitopes_dict: epitopes of this antigen. Keys are indexes of epitopes and values are three letter symbol of epitope's amino acid
            unique_edge_types: Edge types that are available in protein graph. It will be used to one hot encode protein graph's edge types
            
        Returns:
            nx.Graph: A processed NetworkX graph.
        """
        
        if unique_ss is None:
            unique_ss = ['-', 'B', 'G', 'E', 'T', 'S', 'I', 'H']

        if unique_edge_types is None:
            unique_edge_types = ['peptide_bond', 'knn', 'hydrophobic', 'aromatic', 'hbond', 'ionic', 'disulfide']
        
        sequence = g.graph[f"sequence_{chain_id}"]
        for index, (n, d) in enumerate(g.nodes(data=True)):
            assert n.count(":") in [2, 3], f"Might be a corrupted node, check it. Node: {n}"

            assert isinstance(d["coords"], np.ndarray), f"Expected a NumPy array for amino acid coordinates, but got {type(d['coords'])}."

            aa = n.split(":")[1]

            # process asa
            if isinstance(d.get("asa"), pd.core.series.Series):
                asa = d.get("asa")
                asa.dropna(inplace=True)
                asa = list(asa[asa != 0].to_dict().values())
                assert len(asa) == 1, f"Expected object to have length 1 for asa, but got {len(asa)}."
                d["asa"] = asa[0]

            # process rsa
            if isinstance(d.get("rsa"), pd.core.series.Series):
                rsa = d.get("rsa")
                rsa.dropna(inplace=True)
                rsa = list(rsa[rsa != 0].to_dict().values())
                assert len(rsa) == 1, f"Expected object to have length 1 for rsa, but got {len(rsa)}."
                d["rsa"] = rsa[0]

            # process ss
            if isinstance(d.get("ss"), pd.core.series.Series):
                ss = d.get("ss")
                ss.dropna(inplace=True)
                ss = list(ss.unique())
                ss.remove("-") if "-" in ss and len(ss) > 1 else ss
                ss.remove("T") if "T" in ss and len(ss) > 1 else ss
                assert ss, "Empty ss list"
                if len(ss) != 1:
                    raise ValueError(f"Expected object to have length 1 for ss, but got {len(ss)}. Elements: {ss}")
                
                d["ss"] = ss[0]

            # fill empty features
            if not d.get("asa"):
                d["asa"] = 0
            if not d.get("rsa"):
                d["rsa"] = 0.0
            if not d.get("ss"):
                d["ss"] = "-"

            # add physicochemical features  
            for feature in self.amino_acid_physicochemical_features.keys():
                d[feature] = self.amino_acid_physicochemical_features[feature][aa]
            
            # one hot encode secondary structure
            d["ss"] = self.one_hot_encode(classes = [d["ss"]], class_labels=unique_ss)[0].tolist()

            # add epitope annotation
            if self.is_alphafold2_predictions:
                if epitopes_dict.get(str(index)):
                    assert (epitopes_dict[str(index)] == d["residue_name"]), f"Residue name not matching in graph and epitope dataset, Dataset residue name: {epitopes_dict[str(index)]}, Graph residue name: {d['residue_name']}"
                    d["epitope"] = 1
                else:
                    d["epitope"] = 0
            else:
                if epitopes_dict.get(str(d["residue_number"])) and n.count(":") == 2:
                    assert (epitopes_dict[str(d["residue_number"])] == d["residue_name"]), f"Residue name not matching in graph and epitope dataset, Dataset residue name: {epitopes_dict[str(d['residue_number'])]}, Graph residue name: {d['residue_name']}"
                    d["epitope"] = 1
                elif n.count(":") == 3 and epitopes_dict.get("".join(n.split(":")[-2:])):
                    assert (epitopes_dict["".join(n.split(":")[-2:])] == d["residue_name"]), f"Residue name not matching in graph and epitope dataset, Dataset residue name: {epitopes_dict[''.join(n.split(':')[-2:])]}, Graph residue name: {d['residue_name']}"
                    d["epitope"] = 1
                elif n.count(":") == 3 and epitopes_dict.get(str(d["residue_number"])):
                    assert (epitopes_dict[str(d["residue_number"])] == d["residue_name"]), f"Residue name not matching in graph and epitope dataset, Dataset residue name: {epitopes_dict[str(d['residue_number'])]}, Graph residue name: {d['residue_name']}"
                    d["epitope"] = 1
                else:
                    d["epitope"] = 0

            
            # add backbone dihedral angles
            backbone_dihedral_radians = self.calculate_backbone_dihedrals(pdb_path=pdb_path, aa_props=d)
            assert backbone_dihedral_radians, f"Could not find matching amino acid to calculate backbone dihedral angles\nAmino acid:{n}, pdb_path:{pdb_path}"
            d["backbone_dihedral_radians"] = backbone_dihedral_radians
            
            # add sidechain dihedral angles
            sidechain_dihedral_radians = self.calculate_sidechain_dihedrals(pdb_path=pdb_path, aa_props=d)
            assert sidechain_dihedral_radians, f"Could not find matching amino acid to calculate sidechain dihedral angles\nAmino acid:{n}, pdb_path:{pdb_path}"
            d["sidechain_dihedral_radians"] = sidechain_dihedral_radians


            pdbc = f"{os.path.basename(pdb_path).replace('.pdb','')}" if self.is_alphafold2_predictions else f"{os.path.basename(pdb_path).replace('.pdb','')}_{chain_id}"
            # add esm2 amino acid embedding
            embedding = self.esm2_embeddings[pdbc][sequence][index]
            assert embedding.shape[0] == self.embedding_model_to_shape["esm2"], f"Expected embedding shape ({self.embedding_model_to_shape['esm2']},) but got {embedding.shape}"
            assert sequence[index] == RESI_THREE_TO_1[d["residue_name"]], "Sequence mismatch with residue name"
            d["esm2_embedding"] = embedding

            # add esm3 amino acid embedding
            embedding = self.esm3_embeddings[pdbc][sequence][index]
            assert embedding.shape[0] == self.embedding_model_to_shape["esm3"], f"Expected embedding shape ({self.embedding_model_to_shape['esm3']},) but got {embedding.shape}"
            assert sequence[index] == RESI_THREE_TO_1[d["residue_name"]], "Sequence mismatch with residue name"
            d["esm3_embedding"] = embedding

            # add prott5 amino acid embedding
            embedding = self.prott5_embeddings[pdbc][sequence][index]
            assert embedding.shape[0] == self.embedding_model_to_shape["prott5"], f"Expected embedding shape ({self.embedding_model_to_shape['prott5']},) but got {embedding.shape}"
            assert sequence[index] == RESI_THREE_TO_1[d["residue_name"]], "Sequence mismatch with residue name"
            d["prott5_embedding"] = embedding

            # define embedding that will be used to train model
            if self.embedding_model == "esm2":
                d["embedding"] = d["esm2_embedding"]
            elif self.embedding_model == "prott5":
                d["embedding"] = d["prott5_embedding"]
            elif self.embedding_model == "esm3":
                d["embedding"] = d["esm3_embedding"]
            
            self.check_networkx_graph_node_feauture_validity(input_data=d, chain_id=chain_id)

        
        assert len(epitopes_dict) == len([d for _, d in g.nodes(data=True) if d["epitope"] == 1]), f"Epitope count mismatch between epitope dataset and graph. pdb_path: {pdb_path}. chain: {chain_id}"

        for s, t, d in g.edges(data=True):
            edge_type = d["kind"]
            edge_type.remove("knn") if len(edge_type) > 1 and "knn" in edge_type else edge_type
            assert edge_type, "Edge type is not exist"

            d["edge_attr"] = [self.one_hot_encode([_type], unique_edge_types)[0].tolist() for _type in edge_type]
            d["kind"] = edge_type

            for n, n_d in g.nodes(data=True):
                if n == s:
                    source_coords = n_d["coords"]
                elif n == t:
                    target_coords = n_d["coords"]
            
            assert source_coords is not None and target_coords is not None, f"Source or target coordinates does not exist.\nSource coordinates:{source_coords} Target coordinates:{target_coords}"

            d["euclidean_distance"] = round(np.sqrt(np.sum(np.square(source_coords - target_coords))).item(), 5)
            d["source_coords"] = source_coords
            d["target_coords"] = target_coords

            self.check_networkx_graph_edge_feature_validity(input_data=d)

        
        # scale graph
        g = self.scale_graph(g)

        return g

            
    def check_networkx_graph_node_feauture_validity(self, 
                                                    input_data: dict,
                                                    chain_id: str,
                                                    check_list: list[str] = None,
                                                    ) -> None:
        
        """
        Validates the features of a NetworkX graph node against a predefined checklist.
        Parameters:
        -----------
        input_data : dict
            A dictionary containing the node features to be validated.
        chain_id : str
            The expected chain ID to be compared with the node's chain ID.
        check_list : list[str], optional
            A list of keys to be checked in the input_data. If not provided, a default list of keys will be used.
        Raises:
        -------
        Exception
            If any key in the check_list is missing in the input_data.
            If any value in the input_data corresponding to the keys in the check_list is of type pandas Series.
            If the chain_id in the input_data does not match the expected chain_id.
        """

        if check_list is None:
            check_list = [
                'chain_id', 'residue_name', 'residue_number', 'atom_type', 'element_symbol', 'coords', 
                'b_factor', 'amino_acid_one_hot', 'isoelectric_points', 'asa', 'rsa', 'ss', 'Hydropathy', 
                'Volume', 'Chemical', 'Physicochemical', 'Charge', 'Polarity', 'Hydrogen donor or acceptor atom', 
                'backbone_dihedral_radians', 'sidechain_dihedral_radians', 'embedding',
            ]
        for e in check_list:
            if input_data.get(e) is None:
                raise Exception(f"One of the keys is missing. Missing key: {e}, input data: {input_data}")
            
            if isinstance(input_data.get(e), pd.core.series.Series):
                raise Exception(f"Pandas series data type is found for key: {e}")
        
        if input_data["chain_id"] != chain_id:
            raise Exception(f"Expected chain_id is different from networkx graph's chain_id. Graph's chain_id: {input_data['chain_id']}, Expected: {chain_id}")


    def get_node_attributes(self, node_attributes: Union[list[str], Literal["raw", "raw+dihedrals", "raw+dihedrals+embeddings"], None]) -> None:
        """
        Sets the node attributes for the graph.

        Args:
            node_attributes: The node attributes to be included in the graph.

        """
        if node_attributes is None:
            self.node_attributes = [
                'coords', 'b_factor', 'amino_acid_one_hot', 'isoelectric_points', 'asa', 'rsa', 'ss', 'Hydropathy', 
                'Volume', 'Chemical', 'Physicochemical', 'Charge', 'Polarity', 'Hydrogen donor or acceptor atom', 
                'backbone_dihedral_radians', 'sidechain_dihedral_radians', 'embedding',
            ]

        elif isinstance(node_attributes, str):
            if node_attributes not in ["raw", "raw+dihedrals", "raw+dihedrals+embeddings"]:
                raise ValueError(f'Invalid `node_attributes` value: {node_attributes}. Must be from {["raw", "raw+dihedrals", "raw+dihedrals+embeddings"]}.')
            
            if node_attributes == "raw":
                self.node_attributes = [
                'coords', 'b_factor', 'amino_acid_one_hot', 'isoelectric_points', 'asa', 'rsa', 'ss', 'Hydropathy', 
                'Volume', 'Chemical', 'Physicochemical', 'Charge', 'Polarity', 'Hydrogen donor or acceptor atom',
                ]
            elif node_attributes == "raw+dihedrals":
                self.node_attributes = [
                'coords', 'b_factor', 'amino_acid_one_hot', 'isoelectric_points', 'asa', 'rsa', 'ss', 'Hydropathy', 
                'Volume', 'Chemical', 'Physicochemical', 'Charge', 'Polarity', 'Hydrogen donor or acceptor atom', 
                'backbone_dihedral_radians', 'sidechain_dihedral_radians',
                ]
            elif node_attributes == "raw+dihedrals+embeddings":
                self.node_attributes = [
                'coords', 'b_factor', 'amino_acid_one_hot', 'isoelectric_points', 'asa', 'rsa', 'ss', 'Hydropathy', 
                'Volume', 'Chemical', 'Physicochemical', 'Charge', 'Polarity', 'Hydrogen donor or acceptor atom', 
                'backbone_dihedral_radians', 'sidechain_dihedral_radians', 'embedding',
                ]
        
        else:
            self.node_attributes = node_attributes
        


    def check_networkx_graph_edge_feature_validity(self, 
                                                   input_data: dict,
                                                   check_list: list[str] = None) -> None:
        
        if check_list is None:
            check_list = [
                'kind', 'edge_attr', 'euclidean_distance',
            ]
        

        for e in self.edge_attributes:
            if input_data.get(e) is None:
                raise Exception(f"One of the keys is missing. Missing key: {e}, input_data: {input_data}")
            
            if isinstance(input_data.get(e), pd.core.series.Series):
                raise Exception(f"Pandas series data type is found for key: {e}")

    def get_edge_attributes(self, edge_attributes: Union[list[str], None]) -> None:
        """
        Sets the edge attributes for the graph.

        Args:
            edge_attributes: The edge attributes to be included in the graph.

        """
        if edge_attributes is None:
            self.edge_attributes = [
                'kind', 'edge_attr', 'euclidean_distance',
            ]
        else:
            self.edge_attributes = edge_attributes

    def one_hot_encode(self, classes: list, class_labels: list) -> np.array:
        """
        One-hot encodes a list of classes using a specified set of class labels.

        Args:
            classes: A list of classes to encode.
            class_labels: A list of unique class labels.

        Returns:
            A NumPy array of one-hot encoded vectors, where each row represents a class.
        """
        encoding = np.zeros((len(classes), len(class_labels)))
        for i, class_ in enumerate(classes):
            encoding[i, class_labels.index(class_)] = 1
            
        return encoding
    
    def min_max_scaler(self, values: list, index_to_aa: dict[int, str]) -> dict[str, float]:

        """
        Performs min-max scaling on a list of values using list comprehension.
        
        Args:
            values (list): A list of numerical values to be scaled.
            index_to_aa (dict): A dictionary mapping node indices to amino acids.
            
        Returns:
            dict: A dictionary of amino acids mapped to scaled values in the range [0, 1].
        """

        # Find minimum and maximum values
        min_val = min(values)
        max_val = max(values)
        
        # Perform scaling
        scaled_values = [round((val - min_val) / (max_val - min_val), 5) if max_val - min_val != 0 else 0 for val in values]
        
        return {index_to_aa[index]:value for index, value in enumerate(scaled_values)}

    def load_embeddings(self, embedding_path: str, embedding_type: Literal["prott5", "esm2", "esm3"]) -> None:
        """
        Loads embeddings from the specified path and stores them in the corresponding attribute.
        
        Args:
            embedding_path (str): Path to the embedding file.
            embedding_type (str): Type of embedding to load. Should be one of 'prott5', 'esm2', or 'esm3'.
        
        Raises:
            ValueError: If the embedding_type is not one of 'prott5', 'esm2', or 'esm3'.
        """
        if embedding_type not in ["prott5", "esm2", "esm3"]:
            raise ValueError("Invalid embedding type. Please choose 'prott5', 'esm2', or 'esm3'.")
        
        setattr(self, f"{embedding_type}_embeddings", {})
        with h5py.File(embedding_path, "r") as h5_file:
            for grp in h5_file.keys():
                getattr(self, f"{embedding_type}_embeddings")[grp] = {}
                for seq in h5_file[grp].keys():
                    getattr(self, f"{embedding_type}_embeddings")[grp][seq] = np.array(h5_file[grp][seq])

    def get_prott5_embeddings(self) -> None:
        self.load_embeddings(self.prott5_embedding_path, "prott5")

    def get_esm2_embeddings(self) -> None:
        self.load_embeddings(self.esm2_embedding_path, "esm2")
    
    def get_esm3_embeddings(self) -> None:
        self.load_embeddings(self.esm3_embedding_path, "esm3")


    def calculate_backbone_dihedrals(self,
                                    pdb_path: str,
                                    aa_props: dict[str, Any],
                                    to_degree: bool = False,
                                    normalize: bool = True) -> Union[list[float, float, float], None]:
        
        # Load PDB file
        u = mda.Universe(pdb_path)

        # Calculate backbone dihedrals (phi, psi, omega)
        for res in u.residues:
            if res.resid == aa_props["residue_number"] and res.resname == aa_props["residue_name"] and res.segid == aa_props["chain_id"]:
                backbone_dihedrals_dict = {"phi":res.phi_selection(), "psi":res.psi_selection(), "omega":res.omega_selection()}
                # assert not all([True if not v else False for v in backbone_dihedrals_dict.values()]), f"All backbone dihedrals are None.\nResidue: {res.resname}, pdb_path: {pdb_path}"
                if all([True if not v else False for v in backbone_dihedrals_dict.values()]):
                    print(res.resname)
                    print(res.resindex)
                    print(pdb_path)
                    
                backbone_dihedral_radians = []
                for dihedral_selection in backbone_dihedrals_dict.values():
                    if dihedral_selection:
                        coords = [a.position for a in dihedral_selection.atoms]
                        radian = calc_dihedrals(coords[0], coords[1], coords[2], coords[3])
                            
                        if to_degree:
                            value = round(np.degrees(radian).item(), 5)
                        else:
                            value = round(radian.item(), 5)
                        
                        if to_degree and normalize:
                            value = (value - (-180.0)) / (180.0 - (-180.0))
                        elif normalize:
                            value = (value - (-np.pi)) / (np.pi - (-np.pi))
                            
                        backbone_dihedral_radians.append(round(value, 5))
                    else:
                        backbone_dihedral_radians.append(0.0)
                
                return backbone_dihedral_radians

        return None

    def calculate_sidechain_dihedrals(self,
                                        pdb_path: str,
                                        aa_props: dict[str, Any],
                                        to_degree: bool = False,
                                        normalize: bool = True) -> Union[list[float, float, float, float, float], None]:
        
        # Define chi atoms
        # chi_atoms_dict is from: https://gist.github.com/lennax/0f5f65ddbfa278713f58
        chi_atoms_dict = dict(
            chi1 = dict(
                ARG=['N', 'CA', 'CB', 'CG'],
                ASN=['N', 'CA', 'CB', 'CG'],
                ASP=['N', 'CA', 'CB', 'CG'],
                CYS=['N', 'CA', 'CB', 'SG'],
                GLN=['N', 'CA', 'CB', 'CG'],
                GLU=['N', 'CA', 'CB', 'CG'],
                HIS=['N', 'CA', 'CB', 'CG'],
                ILE=['N', 'CA', 'CB', 'CG1'],
                LEU=['N', 'CA', 'CB', 'CG'],
                LYS=['N', 'CA', 'CB', 'CG'],
                MET=['N', 'CA', 'CB', 'CG'],
                PHE=['N', 'CA', 'CB', 'CG'],
                PRO=['N', 'CA', 'CB', 'CG'],
                SER=['N', 'CA', 'CB', 'OG'],
                THR=['N', 'CA', 'CB', 'OG1'],
                TRP=['N', 'CA', 'CB', 'CG'],
                TYR=['N', 'CA', 'CB', 'CG'],
                VAL=['N', 'CA', 'CB', 'CG1'],
            ),
            chi2=dict(
                ARG=['CA', 'CB', 'CG', 'CD'],
                ASN=['CA', 'CB', 'CG', 'OD1'],
                ASP=['CA', 'CB', 'CG', 'OD1'],
                GLN=['CA', 'CB', 'CG', 'CD'],
                GLU=['CA', 'CB', 'CG', 'CD'],
                HIS=['CA', 'CB', 'CG', 'ND1'],
                ILE=['CA', 'CB', 'CG1', 'CD1'],
                LEU=['CA', 'CB', 'CG', 'CD1'],
                LYS=['CA', 'CB', 'CG', 'CD'],
                MET=['CA', 'CB', 'CG', 'SD'],
                PHE=['CA', 'CB', 'CG', 'CD1'],
                PRO=['CA', 'CB', 'CG', 'CD'],
                TRP=['CA', 'CB', 'CG', 'CD1'],
                TYR=['CA', 'CB', 'CG', 'CD1'],
            ),
            chi3=dict(
                    ARG=['CB', 'CG', 'CD', 'NE'],
                    GLN=['CB', 'CG', 'CD', 'OE1'],
                    GLU=['CB', 'CG', 'CD', 'OE1'],
                    LYS=['CB', 'CG', 'CD', 'CE'],
                    MET=['CB', 'CG', 'SD', 'CE'],
            ),
            chi4=dict(
                    ARG=['CG', 'CD', 'NE', 'CZ'],
                    LYS=['CG', 'CD', 'CE', 'NZ'],
            ),
            chi5=dict(
                    ARG=['CD', 'NE', 'CZ', 'NH1'],
            ),
        )
        
        # Load PDB file
        u = mda.Universe(pdb_path)

        # Calculate sidechain dihedrals (chi1, chi2, chi3, chi4, chi5)
        for res in u.residues:
            if res.resid == aa_props["residue_number"] and res.resname == aa_props["residue_name"] and res.segid == aa_props["chain_id"]:
                chi_radians = []
                for chi_res in chi_atoms_dict.values():
                    if chi_res.get(res.resname) and set(chi_res[res.resname]).issubset(set(a.name for a in res.atoms)):
                        chi_selected_atoms = dict.fromkeys(chi_res[res.resname], 1)
                        for a in res.atoms:
                            if chi_selected_atoms.get(a.name) is not None and not isinstance(chi_selected_atoms.get(a.name), np.ndarray):
                                chi_selected_atoms[a.name] = a.position    

                        coords = list(chi_selected_atoms.values())

                        radian = calc_dihedrals(coords[0], coords[1], coords[2], coords[3])
                        if to_degree:
                            value = round(np.degrees(radian).item(), 5)
                        else:
                            value = round(radian.item(), 5)
                        
                        if to_degree and normalize:
                            value = (value - (-180.0)) / (180.0 - (-180.0))
                        elif normalize:
                            value = (value - (-np.pi)) / (np.pi - (-np.pi))
                        
                        chi_radians.append(round(value, 5))
                    else:
                        chi_radians.append(0.0)

                return chi_radians
        
        return None
    def scale_graph(self, g: nx.Graph, scale_attributes: list  = None) -> nx.Graph:
        """
        Performs min-max scaling on node attributes of NetworkX graph.
        
        Args:
            g (nx.Graph): A NetworkX graph to be scaled.
            scale_attributes (list, optional): A list or set of node attributes to be scaled. Defaults to None.
            
        Returns:
            nx.Graph: A NetworkX graph scaled using min-max scaling.
        """
        if scale_attributes is None:
            # Attributes will be scaled
            scale_attributes = ["b_factor", "isoelectric_points", "asa", "rsa"]

        # map node indices to amino acids
        index_to_aa = {index: n for index, n in enumerate(g.nodes(data=False))}

        # Scale node attributes
        for attr in scale_attributes:
            scaled_dict = self.min_max_scaler(values=[d[attr] for _, d in g.nodes(data=True)], index_to_aa=index_to_aa)
            for n, d in g.nodes(data=True):
                d[attr] = scaled_dict[n]
                
        return g

    def create_epitopes_dict_of_antigen(self, antigens_df: pd.DataFrame, pdb_code: str, chain: str) -> dict[str, str]:
        """
        Create a dictionary of epitope indices and three letter amino acids from `antigens_df` and filter according to `file_name`
        Args:
            antigens_df: `pandas.DataFrame` object of antigens which contains antigens' pdb code, chain id, and epitopes
            pdb_code: pdb code of antigen
            chain: chain of antigen 
        Returns:
            a dictionary of epitope indices (key) and three letter amino acids (value)
        """
        epitopes = antigens_df[(antigens_df["PDB"] == pdb_code) & (antigens_df["CHAIN"] == chain)]["Epitopes (resi_resn)"]

        assert not epitopes.empty, "Selected antigen is not found in antigen list"

        epitopes = epitopes.values[0].replace(" ","").split(",")

        return {e.split("_")[0]:e.split("_")[1] for e in epitopes}

    def create_pytorch_geometric_data_object(self, 
                                            g: nx.Graph, 
                                            to_undirected_graph: bool = True
                                        ) -> Data:
        """
        Creates a `torch_geometric.data.Data` object from given `networkx.Graph`
        Args:
            g: `networkx.Graph` object of antigen protein
            to_undirected_graph: If True, the data will be converted to an undirected unidirected graph
        Returns:
            `torch_geometric.data.Data` object of protein graph
        """
        
        node_indexes_mapping = {}
        node_features = collections.defaultdict(list)


        for index, (n, d) in enumerate(g.nodes(data=True)):
            _list = []
            for k, v in d.items():
                if k in self.node_attributes:
                    assert not isinstance(v, str), f"Invalid data type. key:{k}, value: {v}"
                    if isinstance(v, (list, np.ndarray)):
                        _list.extend(list(v))
                    else:
                        _list.append(v)

            node_features["x"].append(_list)

            node_features["pos"].append(d["coords"].tolist())

            node_features["y"].append(d["epitope"])
            
            node_indexes_mapping[n] = index

        
        edge_features = collections.defaultdict(list)
        for s, t, d in g.edges(data=True):
            for index, _ in enumerate(d["kind"]):
                edge_attr = []
                edge_features["edge_index"].append([node_indexes_mapping[s], node_indexes_mapping[t]])
                edge_attr.extend(d["edge_attr"][index])
                edge_attr.append(d["euclidean_distance"])
                edge_features["edge_attr"].append(edge_attr)

        
        data = Data()
        data.x = torch.tensor(node_features["x"], dtype=torch.float)
        data.pos = torch.tensor(node_features["pos"], dtype=torch.float)
        data.y = torch.tensor(node_features["y"], dtype=torch.float)
        data.edge_index = torch.tensor(edge_features["edge_index"], dtype=torch.long).t().contiguous()
        data.edge_attr = torch.tensor(edge_features["edge_attr"], dtype=torch.float)

        if to_undirected_graph:
            data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)
            assert not data.is_directed(), "Data is directed"

        data.validate(raise_on_error=True)

        assert not data.has_isolated_nodes(), "Data object has isolated nodes"

        assert data.edge_attr.shape[0] == data.edge_index.shape[1], "Edge numbers and edge attributes do not match"

        assert g.number_of_nodes() == data.num_nodes, "Node numbers in the graph and Data object do not match"

        return data
    
