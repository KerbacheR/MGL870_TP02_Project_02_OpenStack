# MGL870_TP02_Project_02_OpenStack
MGL870_TP02_Project_02_OpenStack
# Expression régulière pour extraire les informations
log_pattern = re.compile(r'(?P<logfile>[\w\-.]+)\s+(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+(?P<pid>\d+)\s+(?P<level>\w+)\s+(?P<source>[\w.]+)\s+\[(?P<request_id>[\w-]+) (?P<project_id>[\w-]+) (?P<user_id>[\w-]+) [\w-]+\s*\]\s+(?P<client_ip>[\d.]+)\s+"(?P<method>\w+)\s+(?P<url>[\S]+)\s+(?P<http_version>[\S]+)"\s+status:\s+(?P<status>\d+)\s+len:\s+(?P<length>\d+)\s+time:\s+(?P<time>\d+\.\d+)')
