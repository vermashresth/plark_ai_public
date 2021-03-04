from plark_utils.domain_parameters import compute_all_permutations, domain_parameter_ranges
from plark_utils.domain_parameters_json import dump_param_instance_json

endpoint_permutations = compute_all_permutations(domain_parameter_ranges, 
                                                 return_dict=True, endpoints=True)

file_dir = 'endpoint_configs'
for i, config_dict in enumerate(endpoint_permutations):
    dump_param_instance_json(config_dict, file_dir + "/endpoint_config_" + str(i) + ".json")

