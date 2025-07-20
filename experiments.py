import pandas as pd
import numpy as  np
import plots
import faiss_imdb_dbpedia as imdb_dbpedia
import faiss_restaurants as fodors_zagats
import faiss_voters as voters
import faiss_amazon_google as amazon_google
import faiss_amazon_walmart as amazon_walmart
import faiss_abt_buy as abt_buy
import faiss_acm_dblp as acm_dblp
import faiss_scholar_dblp2 as scholar_dblp2



if __name__ == '__main__':
     datasets = [ "IMDB-DBPEDIA",   "AMAZON-WALMART",  "AMAZON-GOOGLE" ,"ABT-BUY", "FODORS-ZAGATS", "ACM-DBLP","SCHOLAR-DBLP", "VOTERS" ]  
     datasets = [ "SCHOLAR-DBLP" ]
     models = ["mini", "mpnet"]
     
     for dataset in datasets:
       for model in models:
         recall_arr= [0]*2
         precision_arr= [0]*2
         model_names=[""]*2 
         phis = [0.15, 0.15]
         for i,suffix in enumerate(["","_ft"]):
           print(f"Working on {dataset} with {model}{suffix}")
           model_name = f"{model}{suffix}"         
           if dataset ==  "IMDB-DBPEDIA":
             print(f"./data/imdb_{model}{suffix}.pqt") 
             df11 = pd.read_parquet(f"./data/imdb_{model}{suffix}.pqt")
             df22 = pd.read_parquet(f"./data/dbpedia_{model}{suffix}.pqt")
             df11['id'] = pd.to_numeric(df11['id'], errors='coerce')
             df22['id'] = pd.to_numeric(df22['id'], errors='coerce')
             df11 = df11.dropna(subset=['title'])
             truth = pd.read_csv("./data/truth_imdb_dbpedia.csv", sep="|", encoding="utf-8", keep_default_na=False)
             valid_d1_ids = set(df11['id'].values)
             valid_d2_ids = set(df22['id'].values)
             mask_to_keep = truth['D1'].isin(valid_d1_ids) & truth['D2'].isin(valid_d2_ids)
             truth = truth[mask_to_keep].copy()          
             ind = imdb_dbpedia
             phi = phis[i]
           elif dataset == "AMAZON-WALMART":
             phis = [0.15, 0.21]
             truth = pd.read_csv("./data/truth_amazon_walmart.tsv", sep="\t", encoding="unicode_escape", keep_default_na=False)
             df22 = pd.read_parquet(f"./data/walmart_products_{model}{suffix}.pqt")
             df11 = pd.read_parquet(f"./data/amazon_products_{model}{suffix}.pqt")
             df11['id'] = pd.to_numeric(df11['id'], errors='coerce')
             df11.dropna(subset=['id'], inplace=True)
             df11['id'] = df11['id'].astype(int)
             df22['id'] = pd.to_numeric(df22['id'], errors='coerce')
             df22.dropna(subset=['id'], inplace=True)
             df22['id'] = df22['id'].astype(int)
             df11.reset_index(drop=True, inplace=True)
             df22.reset_index(drop=True, inplace=True)
             ind = amazon_walmart
             phi = phis[i]
           elif dataset == "ACM-DBLP":
             phis = [0.35, 0.35]
             times = [79,368] 
             truth_file="./data/truth_ACM_DBLP.csv"
             truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
             df22 = pd.read_parquet(f"./data/DBLP_{model}{suffix}.pqt")
             df11 = pd.read_parquet(f"./data/ACM_{model}{suffix}.pqt")
             ind = acm_dblp
             phi = phis[i]
           elif dataset == "FODORS-ZAGATS":
             phis = [0.2, 0.2]
             truth_file="./data/truth_fodors_zagats.csv"
             truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
             df11 = pd.read_parquet(f"./data/fodors_{model}{suffix}.pqt")
             df22 = pd.read_parquet(f"./data/zagats_{model}{suffix}.pqt")
             ind = fodors_zagats
             phi = phis[i]
           elif dataset == "ABT-BUY":    
             phis = [0.15, 0.25]
             truth_file="./data/truth_abt_buy.csv"
             truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
             df11 = pd.read_parquet(f"./data/Abt_{model}{suffix}.pqt")
             df22 = pd.read_parquet(f"./data/Buy_{model}{suffix}.pqt")
             ind = abt_buy
             phi = phis[i]
           elif dataset =="AMAZON-GOOGLE":    
             truth_file="./data/truth_amazon_google.csv"
             truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
             df11 = pd.read_parquet(f"./data/Amazon_{model}{suffix}.pqt")
             df22 = pd.read_parquet(f"./data/Google_{model}{suffix}.pqt")
             ind = amazon_google
             phi = phis[i]
           elif dataset == "SCHOLAR-DBLP":
             times = [306,1175]
             phis = [0.15, 0.20]                
             truth_file="./data/truth_Scholar_DBLP.csv"
             truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
             df22 = pd.read_parquet(f"./data/DBLP2_{model}{suffix}.pqt")
             df11 = pd.read_parquet(f"./data/Scholar_{model}{suffix}.pqt")
             ind = scholar_dblp2
             phi = phis[i]
           elif dataset == "VOTERS":
             truth_file=f"./data/truth_voters.csv"
             truth = pd.read_csv(truth_file, sep=",", encoding="utf-8", keep_default_na=False)
             df11 = pd.read_parquet(f"./data/votersA_{model}.pqt")
             df22 = pd.read_parquet(f"./data/votersB_{model}.pqt")
             ind = voters
             model_name = f"{model}{suffix}"
             phi = phis[i]

       
           recall_arr[i], precision_arr[i] =  ind.run(truth=truth, df11=df11,  df22=df22, model_name=model_name, phi=phi  )
           model_names[i] =  model_name

         results = {
            'model': model_names,
            'recall': recall_arr,
            'precision': precision_arr,
         }
          
         print(results)
         plots_df = pd.DataFrame(results)
         # Calculate F1 Score from recall and precision
         plots_df['f1_score'] = 2 * (plots_df['precision'] * plots_df['recall']) / (plots_df['precision'] + plots_df['recall'])
         plots_df['f1_score'] = plots_df['f1_score'].round(2)
         csv_output_path = f'./plots/results_{dataset}_{model}.csv'
         
         print(f"saving the df to {csv_output_path}")
         plots_df.to_csv(csv_output_path, index=False)
       results_times = {
            'time':times
       }
       times_df = pd.DataFrame(results_times )
       times_df.to_csv(f'./plots/results_{dataset}_times.csv', index=False)


  

