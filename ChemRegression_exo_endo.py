import numpy as np
import pubchempy as pcp
import pandas as pd
import shap
import sklearn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from rdkit.Chem import rdFingerprintGenerator


def load_csv(file):
    data_file=pd.read_csv(file)
    print(data_file.columns)
    return data_file
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


# def compute_descriptors(mol):
#     return [
#         Descriptors.MolWt(mol),
#         Descriptors.TPSA(mol),
#         Descriptors.MolLogP(mol),
#         Descriptors.NumHAcceptors(mol),
#         Descriptors.NumHDonors(mol)
#     ]


def obtain_X_y(data_file):
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


    y=DeltaG_dagger_endo_kcalmol
    plt.hist(y)
    plt.show()

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


    # for diene_sm in dienes_sm:
    #     X_diene_desc.append(compute_descriptors(Chem.MolFromSmiles(diene_sm)))
    # for dienophile_sm in dienophiles_sm:
    #     X_dienophile_desc.append(compute_descriptors(Chem.MolFromSmiles(dienophile_sm)))
    # for solvent_sm in solvents_sm:
    #     X_solvent_desc.append(compute_descriptors(Chem.MolFromSmiles(solvent_sm)))

    X_diene = np.array(X_diene)
    X_dienophile = np.array(X_dienophile)
    X_solvent = np.array(X_solvent)

    X_diene_desc = np.array(X_diene)
    X_dienophile_desc = np.array(X_dienophile)
    X_solvent_desc = np.array(X_solvent)
    #X = np.concatenate([X_diene, X_dienophile, X_solvent, X_dienophile_desc,X_diene_desc,X_solvent_desc,homo_diene.reshape(-1, 1), LUMO_dienophile.reshape(-1, 1),Active_environment_polarity.reshape(-1,1),
     #                   Steric_index.reshape(-1, 1), Electrophilicity_index.reshape(-1, 1)], axis=1)

    X = np.concatenate(
        [X_diene, X_dienophile, X_solvent,  homo_diene.reshape(-1, 1),
         LUMO_dienophile.reshape(-1, 1), Active_environment_polarity.reshape(-1, 1),
                           Steric_index.reshape(-1, 1), Electrophilicity_index.reshape(-1, 1)], axis=1)


    print(X.shape)
    print(y.shape)

    return X,y


def shap_explain(model, X_train_df, X_test_df):
    features = (
            [f"bit{i}-d" for i in range(2048)] +
            [f"bit{i}-dp" for i in range(2048)] +
            [f"bit{i}-s" for i in range(2048)] +
            [
                "homo_diene",
                "LUMO_dienophile",
                "Active_environment_polarity",

                "Steric_index",
                "Electrophilicity_index"
            ]
    )
    X_train_df.columns = features
    X_test_df.columns = features
    explainer = shap.LinearExplainer(model, X_train_df)
    #print(explainer.expected_value)

    shap_values = explainer.shap_values(X_test_df)
    print(features)
    explainer.feature_names=features
    shap.summary_plot(shap_values, X_test_df, max_display=20)


def regression(X,y):
    print(X.shape[1])
    cols = [f"{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=cols)
    #X_df= pd.DataFrame(X)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.30, random_state=999)
    y_train = (y_train)
    y_test = (y_test)
    #
    scalar= StandardScaler()
    X_train_norm = scalar.fit_transform(X_train)
    X_test_norm = scalar.transform(X_test)
    #
    model=LinearRegression()
    #model = SVR(kernel='rbf',C=0.01)
    #model = RandomForestRegressor()
    #model=HistGradientBoostingRegressor()
    #model = Ridge(alpha=1)
    #model=Lasso(alpha=0.01)
    #
    model.fit(X_train_norm, y_train)
    y_test_pred = model.predict(X_test_norm)
    y_train_pred = model.predict(X_train_norm)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test  = r2_score(y_test, y_test_pred)
    mae = sklearn.metrics.mean_absolute_error(y_test, y_test_pred,  sample_weight=None, multioutput='uniform_average')
    r2 = r2_score(y_test, y_test_pred)
    print("Train r2: ", r2_train)
    print("Test r2: ", r2_test)
    print("MAE", mae)
    plt.scatter(y_test.ravel(),y_test_pred.ravel())
    plt.xlabel("Actual endo values")
    plt.ylabel("Predicted endo values")
    plt.title("Actual vs Predicted endo values")
    plt.show()

    y_df = pd.DataFrame({
        "y_true": y_test.ravel(),
        "y_pred": y_test_pred.ravel()
    })
    y_df.to_csv("endo_predicted_y.csv")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

    print(scores)
    print("Mean R2 across 5 folds ",scores.mean())
    print("Std of R2 across 5 folds ",scores.std())



    shap_explain(model, X_train, X_test)

    return r2_train, r2_test



if __name__ == "__main__":
        data_file=load_csv("very_large_data.csv")
        X,y=obtain_X_y(data_file)
        r2_train,r2_test=regression(X,y)



