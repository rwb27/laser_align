

import simplejson as json

config_str = open('config.json').read()

config_data = json.loads(config_str)

print type(config_data['Resolution'])

