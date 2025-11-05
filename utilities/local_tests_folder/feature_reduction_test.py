from sklearn.datasets import load_breast_cancer


class FeatureReduction_Test:

    def __init__(self):
        self.X, self.y = load_breast_cancer(return_X_y=True, as_frame=True)
        self.significance_level = 0.1
        self.correct_forward = [
            "worst concave points",
            "worst radius",
            "worst texture",
            "worst area",
            "smoothness error",
            "worst symmetry",
            "compactness error",
            "radius error",
            "worst fractal dimension",
            "mean compactness",
            "mean concave points",
            "worst concavity",
            "concavity error",
            "area error",
        ]
        self.correct_backward = [
            "mean radius",
            "mean compactness",
            "mean concave points",
            "radius error",
            "smoothness error",
            "concavity error",
            "concave points error",
            "worst radius",
            "worst texture",
            "worst area",
            "worst concavity",
            "worst symmetry",
            "worst fractal dimension",
        ]
