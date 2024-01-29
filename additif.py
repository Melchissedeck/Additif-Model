from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import mean_squared_error

class Additif:
	def __init__(self, data_preparation_object):
		self.data_preparation_object = data_preparation_object
		self.model = LinearRegression()

		self.model.fit(data_preparation_object.x_train, data_preparation_object.y_train)


		y_train_predicted = self.model.predict(data_preparation_object.x_train)
		mean_train_absolute_error = numpy.mean(numpy.abs(y_train_predicted - data_preparation_object.y_train))
		print(f"sur le jeu de train : {mean_train_absolute_error=:.2f}")


		y_test_predicted = self.model.predict(data_preparation_object.x_test)
		mean_test_absolute_error = numpy.mean(numpy.abs(y_test_predicted - data_preparation_object.y_test))
		print(f"sur le jeu de test : {mean_test_absolute_error=:.2f}")

		self.show_model_predictions(y_train_predicted, y_test_predicted)


	def show_model_predictions(self, y_train_predicted, y_test_predicted):
	    plt.figure(figsize=(15, 6))
	    split_index = int(len(self.data_preparation_object.dataset_df) * 0.75)
	   
	    # Aplatir les prédictions et les valeurs réelles si elles ne sont pas déjà 1D
	    y_train_predicted = y_train_predicted.flatten()
	    y_test_predicted = y_test_predicted.flatten()
	    y_train_actual = self.data_preparation_object.y_train.flatten()
	    y_test_actual = self.data_preparation_object.y_test.flatten()
	   
	    # Calcul de l'erreur standard pour les prédictions d'entraînement
	    mse_train = numpy.mean((y_train_actual - y_train_predicted) ** 2)
	    se_train = numpy.sqrt(mse_train) / numpy.sqrt(len(y_train_actual))
	   
	    # Calcul de l'intervalle de confiance pour les prédictions d'entraînement
	    ci_train = 1.96 * se_train
	   
	    # Calcul de l'erreur standard pour les prédictions de test
	    mse_test = numpy.mean((y_test_actual - y_test_predicted) ** 2)
	    se_test = numpy.sqrt(mse_test) / numpy.sqrt(len(y_test_actual))
	   
	    # Calcul de l'intervalle de confiance pour les prédictions de test
	    ci_test = 1.96 * se_test
	   
	    # Tracé des intervalles de confiance
	    # Assurez-vous que la séquence des années est également aplatie
	    
	    years_test = self.data_preparation_object.dataset_df['Years'][split_index:].values.flatten()
	   



	     # Continuation du tracé comme avant
	    plt.plot(
	        self.data_preparation_object.dataset_df['Years'][:split_index],
	        self.data_preparation_object.y_train,
	        "bo-",
	        label='Actual Train Sales'
	    )
	    plt.plot(
	        self.data_preparation_object.dataset_df['Years'][:split_index],
	        y_train_predicted,
	        "indigo",
	        label='Predicted Train Sales'
	    )
	    plt.plot(
	        self.data_preparation_object.dataset_df['Years'][split_index:],
	        self.data_preparation_object.y_test,
	        "ro-",
	        label='Actual Test Sales'
	    )
	    plt.plot(
	        self.data_preparation_object.dataset_df['Years'][split_index:],
	        y_test_predicted,
	        "orange",
	        label='Predicted Test Sales'
	    )

	    plt.fill_between(
	        years_test,
	        y_test_predicted - ci_test,
	        y_test_predicted + ci_test,
	        color='red',
	        alpha=0.2,label = "95% confidence interval"
	    )

	    plt.xlabel("Years")
	    plt.ylabel("Sales")
	    plt.title("Modèle Additif")
	    plt.legend()
	    plt.show()
