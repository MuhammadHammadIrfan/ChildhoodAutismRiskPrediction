from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
autistic_spectrum_disorder_screening_data_for_children = fetch_ucirepo(id=419) 
  
# data (as pandas dataframes) 
X = autistic_spectrum_disorder_screening_data_for_children.data.features 
y = autistic_spectrum_disorder_screening_data_for_children.data.targets 
  
# metadata 
print(autistic_spectrum_disorder_screening_data_for_children.metadata) 
  
# variable information 
print(autistic_spectrum_disorder_screening_data_for_children.variables) 
