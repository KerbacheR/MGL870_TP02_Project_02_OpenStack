# MGL870_TP02_Project_02_OpenStack
MGL870_TP02_Project_02_OpenStack
# Expression régulière pour extraire les informations
log_pattern = re.compile(r'(?P<logfile>[\w\-.]+)\s+(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+(?P<pid>\d+)\s+(?P<level>\w+)\s+(?P<source>[\w.]+)\s+\[(?P<request_id>[\w-]+) (?P<project_id>[\w-]+) (?P<user_id>[\w-]+) [\w-]+\s*\]\s+(?P<client_ip>[\d.]+)\s+"(?P<method>\w+)\s+(?P<url>[\S]+)\s+(?P<http_version>[\S]+)"\s+status:\s+(?P<status>\d+)\s+len:\s+(?P<length>\d+)\s+time:\s+(?P<time>\d+\.\d+)')

# Ligne indiquant q'une erreur s'est produite
nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 03:19:45.356 2931 ERROR oslo_service.periodic_task [req-addc1839-2ed5-4778-b57e-5854eb7b8b09 - - - - -] Error during ComputeManager._run_image_cache_manager_pass
