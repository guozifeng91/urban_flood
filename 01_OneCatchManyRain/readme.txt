rainfall-dependent CNN-based flood prediction model

the portugal and zurich are based on the same cnn architecture

the zurich version changed the data-preprocessing and loss-recording parts, so the code may look nicer.

because i am lazy, the portugal code are left as they are

shared files:
	nn.py
	model/* (pretrained model)

portugal files:
	data_process_portugal.ipynb
	interact_portugal.ipynb (ready to run notebook)
	model_portugal/*
	load_data_portugal/*
	data_portugal/*

zurich files:
	lossPloter.py
	featureExtraction.py
	tensorflow_parser.py