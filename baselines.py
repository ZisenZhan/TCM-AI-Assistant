"""RDKit descriptors and the eight descriptor-based PU baselines."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight


def descriptors(smiles):
    names = [name for name, _ in Descriptors.descList]
    calculator = MolecularDescriptorCalculator(names)
    matrix = np.array([calculator.CalcDescriptors(Chem.MolFromSmiles(s)) for s in smiles])
    return matrix.astype(float), names


class DescriptorFilter:
    def fit(self, matrix):
        x = np.where(np.isfinite(matrix), matrix, np.nan)
        self.nonempty = ~np.isnan(x).all(axis=0)
        x = x[:, self.nonempty]
        self.median = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), self.median, x)
        self.lower, self.upper = np.quantile(x, [.001, .999], axis=0)
        x = np.clip(x, self.lower, self.upper)
        self.variable = x.var(axis=0, ddof=1) > 1e-8
        x = x[:, self.variable]
        if x.shape[1] == 0:
            raise ValueError('No variable descriptors in the training pool.')
        corr = np.nan_to_num(np.abs(np.atleast_2d(np.corrcoef(x, rowvar=False))))
        self.kept = np.array([not np.any(corr[:j, j] > .8) for j in range(x.shape[1])])
        return self

    def transform(self, matrix):
        x = matrix[:, self.nonempty]
        x = np.where(np.isfinite(x), x, self.median)
        x = np.clip(x, self.lower, self.upper)
        return x[:, self.variable][:, self.kept]


def build_models():
    return {
        'GB': GradientBoostingClassifier(n_estimators=150, learning_rate=.05,
                                         max_depth=2, min_samples_leaf=5),
        'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(
            max_depth=2, min_samples_leaf=5, class_weight='balanced'),
            n_estimators=150, learning_rate=.05),
        'NBAYES': GaussianNB(var_smoothing=1e-9),
        'QDA': QuadraticDiscriminantAnalysis(reg_param=.1),
        'SGD logistic': SGDClassifier(loss='log_loss', alpha=1e-4,
            class_weight='balanced', max_iter=3000, tol=1e-4),
        'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'Linear SVM': CalibratedClassifierCV(LinearSVC(C=1., class_weight='balanced',
            dual=False, max_iter=10000), method='sigmoid', cv=5),
        'Logistic regression': LogisticRegression(C=1., class_weight='balanced',
            solver='liblinear', max_iter=2000),
    }


def train_member(matrix, labels, train_ids, test_ids, descriptor_filter):
    train = descriptor_filter.transform(matrix[train_ids])
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(descriptor_filter.transform(matrix[test_ids]))
    models, scores = build_models(), []
    for name, model in models.items():
        kwargs = {'sample_weight': compute_sample_weight('balanced', labels[train_ids])} if name == 'GB' else {}
        model.fit(train, labels[train_ids], **kwargs)
        scores.append(model.predict_proba(test)[:, 1])
    return dict(models=models, descriptor_filter=descriptor_filter, scaler=scaler), np.array(scores)
