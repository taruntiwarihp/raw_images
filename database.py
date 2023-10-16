import pandas as pd
import os

class ManageDataset(object):

    def __init__(self, database_path='database_files/database.csv'):

        self.database_path = database_path

        if os.path.isfile(database_path):
            self.df = pd.read_csv(self.database_path)
        
        else:
            self.df = self.create_database()

    def reload_database(self):
        self.df = pd.read_csv(self.database_path)

    def create_database(self):
        """
            creating dataset if not available 
        """
        print('creating database')
        js = {
            'request_id': [None],
            'time_stamp': [None],
            'doctor_name': [None],
            'patient_name': [None],
            'patient_gender': [None],
            'patient_age': [None],
            'patient_weight':[None],
            'patient_unit': [None],
            'device_focal_length': [None],
            'device_pixel_Pitch': [None],
            'device_type': [None],
            'left_obj': [None],
            'right_obj': [None],
            'left_foot_warp': [None],
            'left_foot_weft': [None],
            'right_foot_warp': [None],
            'right_foot_weft': [None]

        }

        df = pd.DataFrame(js)

        df.drop(index=df.index[0], axis=0, 
                            inplace=True)

        df.to_csv(self.database_path, index=False)

        return df

    def validate_json(self, new_js):
        '''
            params: features as json key and will match 
            return: if keys not matching will return error else none
        '''

        js = {
            'request_id': [None],
            'time_stamp': [None],
            'doctor_name': [None],
            'patient_name': [None],
            'patient_gender': [None],
            'patient_age': [None],
            'patient_weight':[None],
            'patient_unit': [None],
            'device_focal_length': [None],
            'device_pixel_Pitch': [None],
            'device_type': [None],
            'left_obj': [None],
            'right_obj': [None],
            'left_foot_warp': [None],
            'left_foot_weft': [None],
            'right_foot_warp': [None],
            'right_foot_weft': [None]
        }
        if sorted(js.keys()) == sorted(new_js.keys()):
            error = None
        else:
            error = 'Key Names are not matching'
        
        return error

    def update_database(self, js):
        '''
            params: will take json and append at the end of the dataframe 
                    after validation
            return: will return Done if sucessfully, else will return error
        '''
        error = self.validate_json(js)

        if error:
            return error
        
        self.df = self.df.append(js, ignore_index=True)
        self.df.to_csv(self.database_path, index=False)

        return "Done"

    def get_history(self, doc_id, page_no):
        '''
            params: will take doctor id from dataset to get hisrty data
                    along with in page format
            return: will retrun the data entries based on the doctor id
                    else if doc_id not found will return None
        '''
        
        df_out = self.df[self.df['doctor_name'] == doc_id]

        if len(df_out) == 0:

            return "Invalid doctor name"


        elif len(df_out):

            final_df = None
            df_out = df_out.reset_index(drop=True)
            j, k = 0, 0
            for i in range(1, len(df_out) + 1):

                if i % 10 == 0:
                    k += 1
                    
                    if k == page_no:
                        final_df = df_out[j:i]
                        final_df = final_df[:: -1]

                    j = i

                if len(df_out) - 1  == 0:
                    k += 1
                    
                    if k == page_no:
                        final_df = df_out
                    k -= 1
                        

                if len(df_out)  == i:
                    k += 1
                    
                    if k == page_no:
                        final_df = df_out[j:i]
                        final_df = final_df[:: -1]

   

            if final_df is not None:
                final_df.reset_index(drop=True, inplace=True)

                json_data = {}
                json_list = []
                for i in range(len(final_df)):
                    
                    json_list.append(final_df.loc[i].to_dict())
                    # json_data["entry_{}".format(i)] = final_df.loc[i].to_dict()
                
                json_data['data'] = json_list
                
                return json_data

            else:

                return "Invalid page no"

    def get_single_resut(self):
        '''
            return: will return latest entry from dataset
                    if dataset empty return None
        '''

        json_data = {}
        if len(self.df.tail(1)):

            out = self.df.tail(1)
            out.reset_index(drop=True, inplace=True)
            json_data["entry"] = out.loc[0].to_dict()

            return json_data
            
        else:
            return None

    def delete_entries(self, req_id):

        '''
            params: request id which want to delete from dataset
            return: will return Success msg once Done
        '''

        if len(self.df.index[self.df['request_id'] == req_id]):
            self.df.drop(self.df.index[self.df['request_id'] == req_id], inplace= True)
            self.df = self.df.reset_index(drop=True)
            self.df.to_csv(self.database_path, index=False)
            return "success"

        else:
            return None
