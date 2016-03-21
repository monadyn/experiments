import urllib, json
url = "http://csdms.colorado.edu/mediawiki/api.php?action=ask&query=[[Model:%2B]][[Category:Terrestrial||Coastal||Marine||Hydrology||Carbonate||Climate]][[Programming%20language::Fortran77||Fortran90||C||C%2B%2B]][[Source%20code%20availability::Through%20CSDMS%20repository]]|%3FSource%20csdms%20web%20address|limit%3D10000&order%3Ddesc&format=json"
response = urllib.urlopen(url)
data = json.loads(response.read())
print len(data["query"]["results"])
model_names = (data["query"]["results"]).keys()
#print model_names
for key,val in (data["query"]["results"]).items():
	print key,
	print val["printouts"]["Source csdms web address"]


url = "http://csdms.colorado.edu/mediawiki/api.php?action=ask&query=[[Model:%2B]][[Category:Terrestrial||Coastal||Marine||Hydrology||Carbonate||Climate]][[Source%20code%20availability::Through%20CSDMS%20repository]]|%3FSource%20csdms%20web%20address|limit%3D10000&order%3Ddesc&format=json"
response = urllib.urlopen(url)
data = json.loads(response.read())
#print len(data["query"]["results"])
model_names = (data["query"]["results"]).keys()
#print model_names
