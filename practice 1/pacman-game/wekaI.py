import weka.core.jvm as jvm
import weka.core.serialization as serialization
from weka.core.converters import Loader
import weka.core.dataset as ds
from weka.core.dataset import Instances, Attribute, Instance
from weka.classifiers import Classifier

# Para poder utilizar esta clase ejecutar en un terminal los comandos:
# pip install javabridge
# pip install python-weka-wrapper==0.3.0

class Weka:

	# Arranca la maquina virtual de java
	#
	def start_jvm(self):
		jvm.start()

	# Para la maquina virtual de java
	def stop_jvm(self):
		jvm.stop()

	# Predice el valor de la instancia pasada como parametro
	# @param modelName: Nombre del fichero que contiene el modelo generado en weka
	# @param x: La instancia que se pretende clasificar
	# @param arffName: El nombre del fichero arff que se ha utilizado para generar el modelo en Weka
	# @return pred: La clase que predice
	#
	def predict(self, modelName, x, arffName, debug=False):
		# Carga el arrf para conocer la estructura de las instancias
		loader = Loader(classname="weka.core.converters.ArffLoader")
		data = loader.load_file(arffName)

		# Se asume que la clase es el ultimo atributo
		data.class_is_last()

		# Carga del modelo generado en Weka
		objects = serialization.read_all(modelName)
		cls = Classifier(jobject=objects[0])
		if(debug): 
			print("Loaded model...")
			print(cls)

		# Se crea la instancia correspondiente a la entrada y se clasifica
		if(debug): print(("Input", x))

		# Anyade un valor tonto para la clase de la instancia
		if data.class_attribute.is_nominal:
			x.append('a')
		else:
			x.append(0)
		
		# Convierte los valores nominales a la posicion entera que ocupa dentro de sus lista
		for i in range(0, data.num_attributes):
			attribute = data.attribute(i)
			if attribute.is_nominal:
				x[i] = attribute.index_of(x[i]) 			

		# Realiza la prediccion		
		inst = Instance.create_instance(x) 
		inst.dataset = data
		pred = cls.classify_instance(inst)
		if data.class_attribute.is_nominal:
			pred =  data.class_attribute.value(pred)
		if(debug): print(("Prediction", pred))

		return pred

################################# DEBUG ##############################################
#weka = Weka()
#weka.start_jvm()

#x = [1.51793,12.79,3.5,1.12,73.03,0.64,8.77,0,0]
#x = [1.51299,14.4,1.74,1.54,74.55,0,7.59,0,0]
#print(weka.predict("j48.model", x, "./glass.arff"))

#x = ['med','med','5more','more','big','high']
#print(weka.predict("id3.model", x, "./cars.arff"))

#x = ['M',0.455,0.365,0.095,0.514,0.2245,0.101,0.15]
#print(weka.predict("m5p.model", x, "./abalone.arff"))

#weka.stop_jvm()

