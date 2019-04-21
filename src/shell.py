import Datavity

while True:
	text = input('Datavity > ')
	result, error = Datavity.run('<stdin>', text)

	# if error: print(error.as_string())
	
