import Datavity

while True:
    text = input('Datavity > ')

    if text == "exit":
        print("Thank you for using Datavity!")
        break
    elif text == "help":
        print("Datavity: The Feature Engineering Language\n")
        print("\tShow the title of all colums: FEATURE()\n")
        print("\tShows 4 histograms, without change, applying linear scaling, log scaling and clipping: "
              "TRANSFORM(dataFrame)\n")
        print("\tShow first 20 rows of dataFrame: DESCRIBE()\n")
        print("\tExit Datavity: exit\n")
    else:
        result, error = Datavity.run('<stdin>', text)
        if error:
            print(error.as_string())
        elif result:
            print(result)
