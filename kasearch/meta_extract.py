import os
import ast
import json
from joblib import Parallel, delayed
from importlib import resources

import pandas as pd
import numpy as np


class ExtractMetadata:
    def __init__(self, database_path=None, local_oas_path=None):
        
        self._set_id_to_study(database_path, local_oas_path)
    
    def _set_id_to_study(self, database_path, local_oas_path=None):
        """
        Sets id_to_study file.
        """
        
        self.db_path = database_path 
        
        with open(os.path.join(database_path, "id_to_study.txt"), "r") as handle:
            self.id_to_study = ast.literal_eval(handle.readlines()[0])
            
        if local_oas_path:
            for key, val in self.id_to_study.items(): 
                self.id_to_study[key] = val.replace('http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/', local_oas_path)
            
            
    def __group_ids_by_study(self, idxs):
        """
        Groups ids by study for faster extraction by extracting all ids from the same study at the same time.
        """

        idxs_ordered_by_seqid = np.stack([*idxs.T,np.arange(len(idxs))], axis = -1)
        idxs_ordered_by_study = idxs_ordered_by_seqid[idxs_ordered_by_seqid[:, 0].argsort()]
        unique_studies, split = np.unique(idxs_ordered_by_study[:, 0], return_index=True)
        
        return np.split(idxs_ordered_by_study, split[1:])
    
    def _get_single_study_meta(self, idxs):
        """
        Get meta for all ids from a given study.
        """
        study_id, line_ids = idxs[0,0], idxs[:,1]
        study_file = self.id_to_study[study_id]
        
        if "opig.stats.ox.ac.uk" not in study_file: study_file = os.path.join(self.db_path, 'extra_data', study_file)

        if "Bender_2020" in study_file:
            # This is only relevant for versions of OASdb which still include Bender_2020
            print("""Heavy chain data in Bender et al. 2020 has been removed from OAS due to contamination.
Metadata from matches to Bender et al. 2020 sequences therefore return NaN. Either increase 'keep_best_n' 
or use a newer OASdb.""")
            with resources.files(__package__).joinpath('blank_df.csv').open() as blank_df:
                sequence_data = pd.read_csv(blank_df)

            return sequence_data.reindex(list(range(len(line_ids)))).reset_index(drop=True)

        sequence_meta = pd.Series(json.loads(','.join(pd.read_csv(study_file, nrows=0).columns)))
        sequence_data = pd.read_csv(
            study_file, 
            header=1, 
            skiprows=[x+2 for x in range(line_ids.max()) if x not in line_ids], 
            nrows=len(line_ids)
        )        

        for key in sequence_meta.keys():
            sequence_data[key] = sequence_meta[key] 

        sequence_data["rank"] = idxs[np.argsort(line_ids)][:,-1]

        return sequence_data
        
    def _extract_meta(self, idxs, n_jobs=1):
        """
        Extract meta data from all ids for a given query and region.
        """
        
        idxs = [idxs] if any(isinstance(i, int) for i in idxs) else idxs
        
        groups = self.__group_ids_by_study(idxs)
        groups.sort(key=len, reverse=True)
        
        n_groups = len(groups)
        n_jobs = n_groups if n_groups <  n_jobs else n_jobs
        chunksize= n_groups // n_jobs

        fetched_metadata = pd.concat(Parallel(n_jobs=n_jobs)(delayed(self._get_single_study_meta)(group) for group in groups))
        
        return fetched_metadata.sort_values("rank").reset_index(drop=True).drop(columns = ["rank"])
 
 
