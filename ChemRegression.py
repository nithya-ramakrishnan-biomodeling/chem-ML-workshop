import numpy as np
import pubchempy as pcp
import pandas as pd
import sklearn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import  sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

# Get compounds matching the name

data_file=pd.read_csv("very_large_data.csv")
print(data_file.columns)


dienes_sm=data_file['Diene_SMILES'].values
# print(dienes_sm)
dienophiles_sm=data_file['Dienophile_SMILES'].values
# print(dienophiles_sm)
#
solvents_sm=data_file['Solvent_SMILES'].values
homo_diene=data_file['HOMO_diene_eV'].values
LUMO_dienophile=data_file['LUMO_dienophile_eV'].values
Steric_index=data_file['Steric_index'].values
Electrophilicity_index=data_file['Electrophilicity_index'].values
Active_environment_polarity=data_file['Active_environment_polarity'].values
DeltaG_dagger_exo_kcalmol=data_file['DeltaG_dagger_exo_kcalmol'].values
DeltaG_dagger_endo_kcalmol=data_file['DeltaG_dagger_endo_kcalmol'].values






# #print(solvents_sm)
# temp=data_file['Temperature_K'].values
# print(temp)
# exo=data_file['Endo_fraction'].values
# print(exo)

#
y=DeltaG_dagger_endo_kcalmol
plt.hist(y)
plt.show()
# epsilon = 1e-6
# y_clipped = np.clip(y, epsilon, 1 - epsilon)
#
# y_logit = np.log(y_clipped / (1 - y_clipped))
#
from rdkit.Chem import rdFingerprintGenerator

def smiles_to_ecfp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(nBits)

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=nBits
    )

    fp = generator.GetFingerprint(mol)
    return np.array(fp)


def compute_descriptors(mol):
    return [
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol)
    ]
X_diene=[];
X_dienophile=[];
X_solvent=[];
X_diene_desc=[];
X_dienophile_desc=[];
X_solvent_desc=[];
#
#
for diene_sm in dienes_sm:
    X_diene.append(smiles_to_ecfp(diene_sm))
for dienophile_sm in dienophiles_sm:
    X_dienophile.append(smiles_to_ecfp(dienophile_sm))
for solvent_sm in solvents_sm:
    X_solvent.append(smiles_to_ecfp(solvent_sm))

for diene_sm in dienes_sm:
    X_diene_desc.append(compute_descriptors(Chem.MolFromSmiles(diene_sm)))
for dienophile_sm in dienophiles_sm:
    X_dienophile_desc.append(compute_descriptors(Chem.MolFromSmiles(dienophile_sm)))
for solvent_sm in solvents_sm:
    X_solvent_desc.append(compute_descriptors(Chem.MolFromSmiles(solvent_sm)))
#
#
#
#
X_diene = np.array(X_diene)
X_dienophile = np.array(X_dienophile)
X_solvent = np.array(X_solvent)

X_diene_desc = np.array(X_diene)
X_dienophile_desc = np.array(X_dienophile)
X_solvent_desc = np.array(X_solvent)
#
#y= y_logit
#
X = np.concatenate([X_diene, X_dienophile, X_solvent,X_diene_desc,X_dienophile_desc,X_solvent_desc,homo_diene.reshape(-1,1),LUMO_dienophile.reshape(-1,1),Steric_index.reshape(-1,1),Electrophilicity_index.reshape(-1,1),Active_environment_polarity.reshape(-1,1)],axis=1)

# #X = np.concatenate([X_diene, X_dienophile, X_diene_desc,X_dienophile_desc,np.array(temp).reshape(-1,1)],axis=1)
# X = np.concatenate([X_diene, X_dienophile,np.array(temp).reshape(-1,1)],axis=1)
#
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
y_train_log = (y_train)
y_test_log = (y_test)
#
scalar= StandardScaler()
X_train_norm = scalar.fit_transform(X_train)
X_test_norm = scalar.transform(X_test)
#
model=LinearRegression()
#model = SVR(kernel='rbf',C=0.01)
#model = RandomForestRegressor()
#model=HistGradientBoostingRegressor()
# model = Ridge(alpha=1)
# #model=Lasso(alpha=0.01)
#
model.fit(X_train_norm, y_train_log)
y_test_pred = model.predict(X_test_norm)
y_train_pred = model.predict(X_train_norm)
#
#
r2_train = r2_score(y_train_log, y_train_pred)
r2_test  = r2_score(y_test_log, y_test_pred)
r2 = r2_score(y_test_log, y_test_pred)
print(r2_test)
print(r2_train)