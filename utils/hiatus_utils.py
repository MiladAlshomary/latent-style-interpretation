from gradpyent.gradient import Gradient
import json
import csv
import sys
import copy
import os 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import minmax_scale
import random
from data import get_aa_data_from_original_format
from data import get_aa_data_from_original_format
from utils import explanation_interfaces

def get_hrs_data(input_path, num_instances=10, random_seed=123):
    all_df, queries_df, candidates_df = get_aa_data_from_original_format(input_path + '/data/hrs2_09-24-24_english_crossGenre-combined_TA2_input', 
                                                                         input_path + '/groundtruth/hrs2_09-24-24_english_crossGenre-combined_TA2')
    
    #queries_df_sample = queries_df.sample(num_instances, random_state=random_seed)
    queries_df_sample = queries_df[:num_instances]
    for idx, row in queries_df_sample.iterrows():
    
        # Get the query author's documents
        query_author_df   = queries_df[queries_df.authorID == row['authorID']]
        # Get the documents of the equivelant author form the candiates
        corr_author_df    = candidates_df[candidates_df.authorID == row['authorID']]
        # Sample two other authors
        other_two_authors = candidates_df[candidates_df.authorID != row['authorID']].sample(2, random_state=random_seed)
        # Put all the candidate authors together
        all_authors_df    = pd.concat([corr_author_df, other_two_authors])
    
        q_author_ids = [row['authorID']]
        all_authors_df = all_authors_df.groupby('authorID').agg({
            'fullText': lambda x: list(x)
        }).reset_index()
        query_author_df = query_author_df.groupby('authorID').agg({
            'fullText': lambda x: list(x)
        }).reset_index()
        candidate_author_ids = all_authors_df.authorID.tolist()
        ground_truth_assignment = [[1 if x == q_author_ids[0] else 0 for x in all_authors_df.authorID.tolist()]]
        
        yield q_author_ids, candidate_author_ids, query_author_df, all_authors_df, ground_truth_assignment    

def get_iarapa_pilot_data(input_path):
    for data_point in glob.glob(input_path + '*/'):
        candidates_file = list(glob.glob(data_point + '/data/*_candidates.jsonl'))[0]
        queries_file    = list(glob.glob(data_point + '/data/*_queries.jsonl'))[0]
        grount_truth_file = list(glob.glob(data_point + '/groundtruth/*_groundtruth.npy'))[0]
        q_labels_file = glob.glob(data_point + '/groundtruth/*_query-labels.txt')[0]
        c_labels_file = glob.glob(data_point + '/groundtruth/*_candidate-labels.txt')[0]
        
        candidates_df = pd.read_json(candidates_file, lines=True)
        queries_df = pd.read_json(queries_file, lines=True)
        
        queries_df['authorID'] = queries_df.authorIDs.apply(lambda x: x[0])
        candidates_df['authorID'] = candidates_df.authorSetIDs.apply(lambda x: x[0])
        
        queries_df = queries_df.groupby('authorID').agg({'fullText': lambda x: list(x)}).reset_index()
        candidates_df = candidates_df.groupby('authorID').agg({'fullText': lambda x: list(x)}).reset_index()
            
        ground_truth_assignment = np.load(open(grount_truth_file, 'rb'))
        candidate_authors = [a[2:-3] for a in  open(c_labels_file).read().split('\n')][:-1]
        query_authors = [a[2:-3] for a in  open(q_labels_file).read().split('\n')][:-1]
    
        #print(ground_truth_assignment)
        #print(candidate_authors)
        #print(query_authors)
        yield query_authors, candidate_authors, queries_df, candidates_df, ground_truth_assignment

def build_explanation_interface1(explanation_interf, sim_to_styles, style_reps_summ, candidate_authors, random_feature_assignment=False):
    explanation_tmp = copy.copy(explanation_interf)

    if random_feature_assignment:
        random.shuffle(style_reps_summ)
    
    for i, s in enumerate(style_reps_summ):
        if type(s) == list:
            s = ' - '.join(s)
        explanation_tmp = explanation_tmp.replace('[style-{}]'.format(i+1), s)

    for s_id, value in enumerate(style_reps_summ):
        query_author_sim_to_styles = sim_to_styles[0]
        explanation_tmp = explanation_tmp.replace('[query-author-style-{}]'.format(s_id+1), str(round(query_author_sim_to_styles[s_id], 2)))
    
    for order, c_author_id in enumerate(candidate_authors):
            candidate_sim_to_styles = sim_to_styles[order+1] # query author is at 0 position
            for s_id, value in enumerate(candidate_sim_to_styles):
                explanation_tmp = explanation_tmp.replace('[cand-{}-style-{}]'.format(order+1, s_id+1), str(round(value, 2)))
    
    return explanation_tmp

def get_color_gradient(input_list):
    start_color = '#90EE90'
    #start_color = '#ff0000'
    end_color = '#013220'
    #end_color = '#008000'
    # Instantiate the gradient generator, opacity is optional (only used for KML)
    gg = Gradient(gradient_start=start_color, gradient_end=end_color, opacity=1.0)
    return gg.get_gradient_series(series=input_list, fmt='html')

def build_explanation_interface2(explanation_interf, query_author_style_feats, candidate_authors_style_feats, selected_feats, random_feature_assignment=False):
    explanation_tmp = copy.copy(explanation_interf)

    #shuffle features order
    if random_feature_assignment:
        feature_order = list(selected_feats.keys())
        random.shuffle(feature_order)
        selected_feats = {feature_order[i]: item[1] for i, item in enumerate(selected_feats.items())}
    
    #normalize feature weights to scale from 0 to 128   
    feature_weights = [[author[idx] for author in [query_author_style_feats] + candidate_authors_style_feats] for f, idx in selected_feats.items()]
    #log-scale values?
    feature_weights = [np.log(x) for x in feature_weights]
    feature_weights_color = [get_color_gradient(x) for x in feature_weights]
    cell_template = """
        <tr>
            <td style="width: 30%; background-color: rgb(209, 213, 216);">[feat-name]</td>
            <td style="width: 14%; background-color: [author-0-feat-color];"><div style="text-align: center;"><span style="background-color: rgb(239, 239, 239);">[author-0-feat-value]</span></div></td>
            <td style="width: 14%; background-color: [author-1-feat-color];"><div style="text-align: center;"><span style="background-color: rgb(239, 239, 239);">[author-1-feat-value]</span></div></td>
            <td style="width: 14%; background-color: [author-2-feat-color];"><div style="text-align: center;"><span style="background-color: rgb(239, 239, 239);">[author-2-feat-value]</span></div></td>
            <td style="width: 14%; background-color: [author-3-feat-color];"><div style="text-align: center;"><span style="background-color: rgb(239, 239, 239);">[author-3-feat-value]</span></div></td>
        </tr>
    """
    
    table_cells = []
    for feat_order, item in enumerate(selected_feats.items()):
        feat = item[0]
        cell_template_tmp = copy.copy(cell_template)
        cell_template_tmp = cell_template_tmp.replace("[feat-name]", feat)
        for author_order, author_feat_weight in enumerate(feature_weights[feat_order]):
            author_feat_weight_color = feature_weights_color[feat_order][author_order]
            cell_template_tmp = cell_template_tmp.replace("[author-{}-feat-color]".format(author_order), author_feat_weight_color)
            #cell_template_tmp = cell_template_tmp.replace("[author-{}-feat-value]".format(author_order), str(round(author_feat_weight,2)))
            cell_template_tmp = cell_template_tmp.replace("[author-{}-feat-value]".format(author_order), '')
        

        table_cells.append(cell_template_tmp)

    explanation_tmp = explanation_tmp.replace("[table-body]", "\n".join(table_cells))
    
    return explanation_tmp