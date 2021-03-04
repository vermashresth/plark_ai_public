from plark_utils.domain_parameters import generate_random_param_instance
from plark_utils.domain_parameters_json import dump_param_instance_json

rand_params = generate_random_param_instance()
print(rand_params)

dump_param_instance_json(rand_params, 'random_instance.json')

