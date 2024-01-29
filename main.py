from data_preparation import DataPreparation
from additif import Additif


csv_path = "vente_maillots_de_bain(1).csv"
data_preparation_object = DataPreparation(csv_path)
additif_object = Additif(data_preparation_object)

data_preparation_object.display_dataframe()
#data_preparation_object.show_graph()

# Fait par  HAMZA LAZTOUTI  ET Johannes AFOUDAH
