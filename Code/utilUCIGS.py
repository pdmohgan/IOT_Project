import requests
import zipfile
import os
import shutil
import sys

def ifFileExists(fname):
	return os.path.exists(fname)

def deleteHeaderFromFile(fname):
	f=open(fname)
	output=[]
	
	for line in f:
		if not "id" in line:
			output.append(line)
	
	f.close()
	f=open(fname,'w')
	f.writelines(output)
	f.close()

def dnldAndPreprocessData(datadir):
	file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00362/HT_Sensor_UCIsubmission.zip"
	sys.stdout.write("Downloading Dataset from uci site\n ")
	
	r=requests.get(file_url,stream=True)
	with open(datadir + "data.zip","wb") as zip:
		for chunk in r.iter_content(chunk_size=1024):
			
			#writing one chunk at a time to zip file
			if chunk:
				zip.write(chunk)
				sys.stdout.write("#")
	print("\nDataset Download Completed!")
	print("Extracting Dataset")
		
	with zipfile.ZipFile(datadir + "data.zip","r") as zip_ref:
		zip_ref.extractall(datadir)
	with zipfile.ZipFile(datadir + "HT_Sensor_dataset.zip","r") as zip_ref:
		zip_ref.extractall(datadir)

	print("Removing unwanted files")
	os.remove(datadir + "data.zip")
	os.remove(datadir + "HT_Sensor_dataset.zip")
	shutil.rmtree(datadir + "__MACOSX",ignore_errors=True)

	print("Preprocessing Dataset")
	deleteHeaderFromFile(datadir + "HT_Sensor_dataset.dat")
	deleteHeaderFromFile(datadir + "HT_Sensor_metadata.dat")
