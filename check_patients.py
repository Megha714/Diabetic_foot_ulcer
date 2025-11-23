import database
from bson import json_util
import json

db = database.get_db()
patients_col = db['patients']

print('Total patients:', patients_col.count_documents({}))
print('\nPatient documents:')
for p in patients_col.find().limit(3):
    print(json.dumps(p, default=json_util.default, indent=2))
