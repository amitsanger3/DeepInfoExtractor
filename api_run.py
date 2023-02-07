import os, sys
from api import app


os.system("set FLASK_APP=LMR_api")
app.run(host="0.0.0.0", port="5000", debug=True)
