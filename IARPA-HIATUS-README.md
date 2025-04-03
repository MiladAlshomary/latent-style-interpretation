# Style Generation Pipeline


## Generating Styles Corpus

Example usage for generating styles:
```
python generate_styles.py --data-dir datasets/example_data.jsonl 
                          --generator-model llama3 
                          --shortener-model openai:gpt-3.5-turbo
                          --style-threshold 2
                          --max-new-tokens 512 
                          --device 2 
```

The output of this command would generate "filtered/refined_and_aggregated_features_final.csv" inside the data-dir where each row contains a writing style feature and the corresponding documentID it appears in.

### For IARPA-HIATUS

We ran this command to generate writing style for our HRS corpus using DeepSeek and the results can be explored in src-ipynb/generate_styles_w_deepseek.ipynb

## Clustering and creating the interpretable space

Example usage for clustering:
```
python cluster_documents.py --train-dir datasets/example_data.jsonl 
                            --test-dir datasets/example_data.jsonl 
                            --save-dir results/
```
`cluster_documents.py` must be run after `generate_styles.py` to construct the interpretable space, where each basis is mapped to a style distribution.

### For IARPA-HIATUS

To run the code for clustering the HRS data to interpret the latent space of LUAR model:

- Command to explore different EPS parameters of DBSCAN, it prints at the end a list of EPS values and the correpsonding quality measures

``` LORA_BASEMODEL_CHECKPOINT_PATH="/data_folder/milad/hiatus-data/models/ta2-system2-en-submitted-8-01-25/rrivera1849/LUAR-MUD/" python cluster_documents.py --train-dir "/data_folder/milad/hiatus-data/explainability_all_data/train_authors.json" --test-dir "/data_folder/milad/hiatus-data/explainability_all_data/test_authors.json" --save-dir "/data_folder/milad/hiatus-data/explainability_all_data/luar_clusters/" --model  'luar-mud' --style-dir "/data_folder/milad/hiatus-data/explainability_all_data/refined_and_aggregated_features_final.csv" --style_feat_column 'final_attribute_name' --top_k_feats 10```

- This command will then create the clusterings for the EPS value 0.09 (the best I have found to be)

LORA_BASEMODEL_CHECKPOINT_PATH="/data_folder/milad/hiatus-data/models/ta2-system2-en-submitted-8-01-25/rrivera1849/LUAR-MUD/" python cluster_documents.py --train-dir "/data_folder/milad/hiatus-data/explainability_all_data/train_authors.json" --test-dir "/data_folder/milad/hiatus-data/explainability_all_data/test_authors.json" --save-dir "/data_folder/milad/hiatus-data/explainability_all_data/luar_clusters_09/" --model  'luar-mud' --style-dir "/data_folder/milad/hiatus-data/explainability_all_data/refined_and_aggregated_features_final.csv" --style_feat_column 'final_attribute_name' --top_k_feats 10 --eps 0.09

The output would be a pickle file under ```/data_folder/milad/hiatus-data/explainability_all_data/luar_clusters_09/``` called interpretable_space.pkl containing the interpretable space (latent representation and distribution over writing style feature of every dimension)

- Although the cluster style distribution is computed in the clustering code, we run additional code to generate different style representations for every cluster inside the generate_interp_space_style_rep notebook. The function ```build_cluster_representation``` requires the path to interpretable space (e.g: ```/data_folder/milad/hiatus-data/explainability_all_data/train_authors.pkl```) and the style corpus (e.g: ```/data_folder/milad/hiatus-data/explainability_all_data/interpretable_space_representations.json```)

## Generating Explanations

Example usage for generating style explanations:
```
python generate_explanations.py --inter-space <path to the interpretable space generated from cluster_documents.py code> 
                                --input-path <path to the dataframe containing the documents to be explained> 
                                --output-path <path to where to save the output>

```
`cluster_documents.py` must be run after `generate_styles.py` to construct the interpretable space, where each basis is mapped to a style distribution.

### For IARPA-HIATUS explanations
Now, given the interpretable_space.pkl and its style representation interpretable_space_representations.json, generated previsously, the notebook generate-iarpa-explanations notebook contains the code to generate explanations for HRS examples.


## Requirements

Create your conda enviornment from gpu-env-file.txt:
```
conda create --name myenv --file gpu-env-file.txt
```

Install the necessary libraries using the provided `gpu-env-pip-requirements.txt`:
```
pip install -r gpu-env-pip-requirements.txt
```