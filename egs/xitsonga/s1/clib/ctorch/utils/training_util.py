def continue_train(message=""):
    """ Returns a bool indicating continue training or not and an integer of how more epochs to train"""
    continue_or_not = ""
    while continue_or_not not in {'yes', 'y', 'no', 'n'}:
        continue_or_not = input(message + "Continue to train [y/n]?").lower().strip()

    add_epochs = "0" if continue_or_not in {'no', 'n'} else ""
    while not add_epochs.isdigit():
        add_epochs = input("How many addition epochs [1 to N]:").lower().strip()

    return continue_or_not in {'yes', 'y'}, int(add_epochs)
