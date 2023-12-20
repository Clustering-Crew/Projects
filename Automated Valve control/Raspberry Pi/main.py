import sklearn
import joblib
import thingspeak
from urllib.request import urlopen
import json
import pandas as pd


WRITE_API = "GR19NG95R9OSJT5U"
READ_API = "NQ08NQEDCSEBD2D4"
CHANNEL_ID = 2266425
WRITE_CHANNEL = 2288225

channel = thingspeak.Channel(WRITE_CHANNEL, WRITE_API)

while True:
	TS = urlopen("http://api.thingspeak.com/channels/%s/feeds/last.json?api_key=%s" % (CHANNEL_ID,READ_API))
	response = TS.read()
	data=json.loads(response)
	print(data['field1'])
	print(data['field3'])

	model = joblib.load('model.pkl')	
	test_df = pd.DataFrame({'temperature': [data['field1']], 'soilmiosture': [data['field3']]})
	pred = model.predict(test_df)
	print(pred)
	channel.update({'field1': pred[0]})	
