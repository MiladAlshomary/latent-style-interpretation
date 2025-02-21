import os
import argparse
import warnings
import pickle as pkl
from collections import Counter, defaultdict
import spacy

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

from data import get_aa_data
from utils import find_first_minimal_change, safe_parse
from metrics import compute_model_performance
from aa_models import get_model
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
from scipy.stats import zscore



warnings.filterwarnings("ignore")


def main(args):
    """
    Main function to generate explanations for documents based on a given cluster-based interpretable space

    Parameters:
    args (dict): Dictionary of arguments including:
        --inter-space (str): path to the dataframe representing the interpretable space.
        --input-path (str): Path to the documents dataframe
        --output-path (str): Path to the output
        --top-k (int): Number of top features to take from the cluster
        --top-c (int): Number of top dimensions to take

    Steps:
    ------
    1. Generate the latent embeddings of all documents
    2. Project documents using the corresponding proj_matrix of the interpretable space
    3. Take the top-k features of the top-c dimensions (clusters) as the style explanations of the document 
    
    Example usage:
    --------------
    python generate_explanations.py --inter-space <path to the interpretable space generated from cluster_documents.py code> 
                                    --input-path <path to the dataframe containing the documents to be explained> 
                                    --output-path <path to where to save the output>
    
    """

    model = get_model(args["model"])

    
    # Load model and data
    #'../data/explainability/explainability_experiment_test_ds.pkl'
    documents_df = pd.read_json(args['input_path'])
    
    #'../data/explainability/clusterd_authors_with_style_description.pkl'
    interpretable_space = pkl.load(open(args['inter_path'], 'rb'))
    del interpretable_space[-1] #DBSCAN generate a cluster -1 of all outliers. We don't want this cluster
    #print("# clusters:", len(interpretable_space))
    dimension_to_latent = {key: interpretable_space[key][0] for key in interpretable_space}
    dimension_to_style  = {key: interpretable_space[key][1] for key in interpretable_space}

    proj_matrix = np.array(list(dimension_to_latent.values()))
    #print(proj_matrix)
    #normalize projection matrix
    proj_matrix = normalize(proj_matrix, axis=1, norm='l2')

    documents_assigned_clusters, documents_ranked_clusters = document_to_cluster_assignment(model, proj_matrix, documents_df.fullText.tolist())

    # Aggregate the top-k fetures of the top-n clusters to be the final list
    final_documents_reps = []
    for i, ranked_clusters in enumerate(documents_ranked_clusters):
        rep_feats = []
        for cluster_id in ranked_clusters[:args['top_c']]:
            cluster_feats = sorted(dimension_to_style[cluster_id].items(), key=lambda x: -x[1])
            #print(cluster_id)
            #print(cluster_feats)
            rep_feats+= [x[0] for x in cluster_feats[:args['top_k']]]
        final_documents_reps.append(rep_feats)

    documents_df['documetn_style_description'] = final_documents_reps
    documents_df.to_json(args['output_path'])

def get_documents_style_descriptions(documents, model_path, interp_space_path, interp_space_rep_path, style_feat_clm, top_c=3, top_k=10, flip_cluster_order=False):
    
    model = get_model(model_path)
    
    #'../data/explainability/clusterd_authors_with_style_description.pkl'
    interpretable_space = pkl.load(open(interp_space_path, 'rb'))
    del interpretable_space[-1] #DBSCAN generate a cluster -1 of all outliers. We don't want this cluster
    #print("# clusters:", len(interpretable_space))
    dimension_to_latent = {key: interpretable_space[key][0] for key in interpretable_space}
    
    #Load interp space representations
    interpretable_space_rep_df = pd.read_json(interp_space_rep_path)
    dimension_to_style  = {x[0]: x[1] for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[style_feat_clm].tolist())}

    
    proj_matrix = np.array(list(dimension_to_latent.values()))
    #print(proj_matrix)
    #normalize projection matrix
    proj_matrix = normalize(proj_matrix, axis=1, norm='l2')
    
    documents_assigned_clusters, documents_ranked_clusters = document_to_cluster_assignment(model, proj_matrix, documents)

    #print([r[0] for r in documents_ranked_clusters])
    
    # Aggregate the top-k fetures of the top-n clusters to be the final list
    final_documents_reps = []
    final_documents_clusters = []
    final_documents_dist_rep = []
    for i, ranked_clusters in enumerate(documents_ranked_clusters):
        rep_feats = []
        if flip_cluster_order:
            ranked_clusters = list(reversed(ranked_clusters))

        for cluster_id in ranked_clusters[:top_c]:
            cluster_feats = dimension_to_style[cluster_id]
            rep_feats+= [(cluster_id, x) for x in cluster_feats[:top_k]]
        final_documents_reps.append(rep_feats)
        final_documents_clusters.append([(c, documents_assigned_clusters[i][c]) for c in ranked_clusters[:top_c]])
    
    return final_documents_reps, final_documents_clusters


def get_documents_rep_vectors(documents, model_path, interp_space_path):
    model = get_model(model_path)
    
    interpretable_space = pkl.load(open(interp_space_path, 'rb'))
    del interpretable_space[-1] #DBSCAN generate a cluster -1 of all outliers. We don't want this cluster
    #print("# clusters:", len(interpretable_space))

    
    dimension_to_latent = {key: interpretable_space[key][0] for key in interpretable_space}
    proj_matrix = np.array(list(dimension_to_latent.values()))
    proj_matrix = normalize(proj_matrix, axis=1, norm='l2')

    # Compute latent and interpretable vectors for query and candidate documents
    documents_latent = model.encode(documents)
    documents_interp  = [proj_matrix.dot(e) for e in documents_latent]

    return documents_latent, documents_interp
    
def get_explainable_clusters(query_interp, candidates_interps, predicted_author_idx=None, num_clusters=3, method=1):

    if method == 1:
        return np.argsort(query_interp)[::-1][:num_clusters]
    
    elif method == 2:
        # Here, we look at how similar each candidate to the query wrt each cluster (query_doc - x) and then compute a z-score
        # reflecting how different this similarity compared to the other candidates, and then take the clusters that distinguish the 
        # predicted author from the other candidates by being much closer to the query author.
        authors_diffs = [abs(query_interp - x) for x in candidates_interps]
        res = zscore(authors_diffs, axis=0)
        return np.argsort(res[predicted_author_idx])[:num_clusters]
    
def explain_model_prediction_over_author(model_path, inter_space_path, inter_space_rep_path, query_author, candidate_authors, top_c=3, top_k=5, style_feat_clm='tfidf_rep_5', cluster_lvl=True, style_feat_summary_clm=None):

    #load ta2 model
    model = get_model(model_path)
    
    #load interpretable_space and compute projection matrix
    interpretable_space = pkl.load(open(inter_space_path, 'rb'))

    #Load interp space representations
    interpretable_space_rep_df = pd.read_json(inter_space_rep_path)
    dimension_to_style  = {x[0]: x[1] for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[style_feat_clm].tolist())}
    all_style_feats = list(set([f for row in interpretable_space_rep_df[style_feat_clm].tolist() for f in row]))

    if style_feat_summary_clm == None:
        dimension_to_style_summary  = {x[0]: ' - '.join(x[1]) for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[style_feat_clm].tolist())}
    else:
        dimension_to_style_summary  = {x[0]: x[1] for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[style_feat_summary_clm].tolist())}
    
    del interpretable_space[-1] #DBSCAN generate a cluster -1 of all outliers. We don't want this cluster
    #print("# clusters:", len(interpretable_space))
    dimension_to_latent = {key: interpretable_space[key][0] for key in interpretable_space}
    
    proj_matrix = np.array(list(dimension_to_latent.values()))
    proj_matrix = normalize(proj_matrix, axis=1, norm='l2')

    # Compute latent and interpretable vectors for query and candidate authors
    q_author_latents = model.encode(query_author)
    q_author_interps = [proj_matrix.dot(e/np.linalg.norm(e)) for e in q_author_latents]
    
    c_author_latents = [model.encode(c_author) for c_author in candidate_authors]
    c_author_interps = [[proj_matrix.dot(e/np.linalg.norm(e)) for e in documents_latent] for documents_latent in c_author_latents]

    # Convert cosine similarity to angular similarity (ranges from 0 to 1)
    q_author_interps = [1 - (np.arccos(cos_sim) / np.pi) for cos_sim in q_author_interps]
    c_author_interps = [[1- (np.arccos(cos_sim) / np.pi) for cos_sim in c_author_interp] for c_author_interp in c_author_interps]
    
    # Author representations as an average of their documents
    q_author_latent_avg  = np.mean(q_author_latents, axis=0)
    q_author_interp_avg  = np.mean(q_author_interps, axis=0)
    c_author_latent_avgs = [np.mean(x, axis=0) for x in c_author_latents]
    c_author_interp_avgs = [np.mean(x, axis=0) for x in c_author_interps]
    
    
    
    #find most representative document of the query author
    q_cos_sim = cosine_similarity([q_author_latent_avg], q_author_latents)
    q_author_rep_document = np.argmax(q_cos_sim)

    #find most representative document of the candidate authors
    c_cos_sims = [cosine_similarity([c_author[0]], c_author[1]) for c_author in zip(c_author_latent_avgs, c_author_latents)]
    c_author_rep_documents = [np.argmax(x) for x in c_cos_sims]
    

    # Option 1: Compute Model's latent and interp prediction as the pairwise cosine similarity over their documents' representations
    # latent_similarities = [np.mean(cosine_similarity(q_author_latents, c_latent)) for c_latent in c_author_latents]
    # interp_similarities = [np.mean(cosine_similarity(q_author_interps, c_interp)) for c_interp in c_author_interps]
    # model_latent_rank = np.argsort(latent_similarities)[::-1]
    # model_interp_rank = np.argsort(interp_similarities)[::-1]

    # Option 2: Compute Model's latent and interp prediction as the cosine similarity over their most representative documents
    # latent_similarities = [np.mean(cosine_similarity([q_author_latents[q_author_rep_document]], [c_latent[c_author_rep_documents[i]]])) for i, c_latent in enumerate(c_author_latents)]
    # interp_similarities = [np.mean(cosine_similarity([q_author_interps[q_author_rep_document]], [c_interp[c_author_rep_documents[i]]])) for i, c_interp in enumerate(c_author_interps)]
    # model_latent_rank = np.argsort(latent_similarities)[::-1]
    # model_interp_rank = np.argsort(interp_similarities)[::-1]
    
    # Option 3: Compute Model's latent and interp prediction as the cosine similarity over their average representations
    latent_similarities = cosine_similarity([q_author_latent_avg], c_author_latent_avgs)[0]
    interp_similarities = cosine_similarity([q_author_interp_avg], c_author_interp_avgs)[0]
    model_latent_rank = np.argsort(latent_similarities)[::-1]
    model_interp_rank = np.argsort(interp_similarities)[::-1]
    
    # print(latent_similarities, interp_similarities)
    # print(model_latent_rank, model_interp_rank)
    # print()
    
    #### Explanation #########

    #q_rep_document_interp = q_author_interps[q_author_rep_document]
    #c_rep_document_interp = [c_author_interps[i][idx] for i, idx in enumerate(c_author_rep_documents)]
    q_rep_document_interp = q_author_interp_avg
    c_rep_document_interp = c_author_interp_avgs

    predicted_author = model_interp_rank[0]
    
    # For explainability we rescale the cosine similarity from [-1,+1] to [0,2]
    # q_rep_document_interp = (q_rep_document_interp + 1)/2
    # c_rep_document_interp = [(x + 1)/2 for x in c_rep_document_interp]
        
    if cluster_lvl:        
        # Find cluster assignment for the query and candidate documents
        query_cluster_assignments = q_rep_document_interp.tolist()
        #query_cluster_rankings    = np.argsort(query_cluster_assignments)[::-1]
        
        candidate_cluster_assignments = [interp.tolist() for interp in c_rep_document_interp]
        #candidate_cluster_rankings    = [np.argsort(ass)[::-1] for ass in candidate_cluster_assignments]

        # Find the clusters to present as explanations, either method 1 (top similar clusters to the query) or method 2 (using z-score)
        rep_clusters = get_explainable_clusters(q_rep_document_interp, c_rep_document_interp, predicted_author, num_clusters=3, method=2)
        #print(rep_clusters)
        
        cluster_style_reps = [dimension_to_style[cidx][:top_k] for cidx in rep_clusters]
        cluster_style_reps_summ = [dimension_to_style_summary[cidx] for cidx in rep_clusters]
        
        # Compute how similar the candidate documents to these top_c clusters
        author_distance_to_clusters = [[author[cidx] for cidx in rep_clusters]
            for author in  [query_cluster_assignments] + candidate_cluster_assignments]

        # for i, arr in enumerate(candidates_distance_to_clusters):
        #     print(arr)
        
        return model_latent_rank, model_interp_rank, cluster_style_reps, cluster_style_reps_summ, author_distance_to_clusters, q_author_rep_document, c_author_rep_documents
    else:
        # Get a ranked list of style features for all documents based on thir similarity to clusters
        documents_feats, selected_feats_idxs = find_ranked_style_feats_for_documents([q_rep_document_interp] + c_rep_document_interp, dimension_to_style, all_style_feats, predicted_author)

        selected_feats = {all_style_feats[f_idx]: f_idx for f_idx in selected_feats_idxs}
        return model_latent_rank, model_interp_rank, documents_feats[0],  documents_feats[1:], selected_feats,  q_author_rep_document, c_author_rep_documents
    

def find_ranked_style_feats_for_documents(document_interps, dimension_to_style, style_feats, predicted_author_idx, num_feats=10):
    documents_cluster_assignments = [interp.tolist() for interp in document_interps]
    
    documents_feats_weights = []
    for document_cluster_assignments in documents_cluster_assignments:
        feature_weights = [np.mean([(cluster_sim * dimension_to_style[cluster_id][f]) if f in dimension_to_style[cluster_id] else 0 for cluster_id, cluster_sim in enumerate(document_cluster_assignments)]) for f in style_feats]
        #documents_feats.append({x[0]: x[1] for x in zip(style_feats, feature_weights)})
        documents_feats_weights.append(np.array(feature_weights))

    query_feats = documents_feats_weights[0]
    # Find features that distinguish predicted author from other candidates
    authors_diffs = np.array([abs(query_feats - x) for x in documents_feats_weights[1:]])
    authors_diffs = authors_diffs[:,0,:] if len(authors_diffs.shape) > 2 else authors_diffs # for some reason we are getting an array of 3 dimensions
    res = zscore(authors_diffs, axis=0)
    selected_feats = np.argsort(res[predicted_author_idx])[:num_feats]
    return documents_feats_weights, selected_feats
            
    
def document_to_cluster_assignment(model, proj_matrix, documents):

    documents_latent = model.encode(documents)
    documents_interp  = [proj_matrix.dot(e) for e in documents_latent]

    cluster_scores  = [x.tolist() for x in documents_interp]
    ranked_clusters = [list(np.argsort(x)[::-1]) for x in cluster_scores]

    return cluster_scores, ranked_clusters
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--inter-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--top-c", type=int, required=True)
    args = vars(parser.parse_args())
    print(args)
    main(args)
