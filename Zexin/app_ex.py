import argparse

parser = argparse.ArgumentParser()
parser.add_argument("Nom",help = "Nom d'Entreprise",type = str)
parser.add_argument("Date_debut",help = "Date de debut de twitters, form = \"YYYYMMDD\" ",type = str)
parser.add_argument("Date_fin",help = "Date de fin de twitters, form = \"YYYYMMDD\" ",type = str)

args = parser.parse_args()

print(args.Nom)
print(args.Date_debut)
print(args.Date_fin)

