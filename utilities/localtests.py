import unittest

import numpy as np
from audiocompression import AudioCompression
from eigenfaces import Eigenfaces
from feature_reduction import FeatureReduction
from logistic_regression import LogisticRegression, hyperparameter_tuning
from regression import Regression
from smote import SMOTE
from svd_recommender import SVDRecommender
from utilities.local_tests_folder.ac_test import AC_Test
from utilities.local_tests_folder.ef_test import EF_Test
from utilities.local_tests_folder.feature_reduction_test import FeatureReduction_Test
from utilities.local_tests_folder.lr_test import LogisticRegression_Test
from utilities.local_tests_folder.regression_test import Regression_Test
from utilities.local_tests_folder.smote_test import SMOTE_Test
from utilities.local_tests_folder.svd_recommender_test import SVDRecommender_Test


def print_success_message(msg):
    print(f'UnitTest passed successfully for "{msg}"!')


class TestEigenfaces(unittest.TestCase):
    """
    Tests for Eigenfaces
    """

    def test_svd(self):
        """
        Test correct implementation of SVD calculation for eigenfaces
        """
        ef = Eigenfaces()
        test_ef = EF_Test()
        try:
            U, S, V = ef.svd(test_ef.faces)
        except MemoryError:
            self.assertTrue(False, "Your solution is allocating too much memory.")
        self.assertEqual(U.shape, test_ef.shape_of_U, "Shape of U is incorrect")
        self.assertEqual(S.shape, test_ef.shape_of_S, "Shape of S is incorrect")
        self.assertEqual(V.shape, test_ef.shape_of_V, "Shape of V is incorrect")
        reconstructed = U @ np.diag(S) @ V
        self.assertTrue(
            np.allclose(reconstructed, test_ef.faces - np.mean(test_ef.faces, axis=0)),
            "Your SVD does not reconstruct the centered faces. Did you remember to center the data?",
        )
        success_msg = "SVD calculation"
        print_success_message(success_msg)

    def test_compute_eigenfaces(self):
        ef = Eigenfaces()
        test_ef = EF_Test()
        eigenfaces = ef.compute_eigenfaces(test_ef.faces, 2)
        self.assertEqual(
            eigenfaces.shape,
            test_ef.eigenfaces_shape,
            "Eigenfaces matrix shape is incorrect",
        )
        self.assertTrue(
            np.allclose(np.sum(eigenfaces), test_ef.eigenfaces_sum),
            "Eigenfaces are incorrect",
        )
        success_msg = "Eigenfaces"
        print_success_message(success_msg)

    def test_project(self):
        ef = Eigenfaces()
        test_ef = EF_Test()
        test_eigenfaces = test_ef.faces[0:2]
        test_faces = test_ef.faces[2:5]
        projections = ef.project(test_faces, test_eigenfaces)
        self.assertEqual(
            projections.shape,
            (3, 2),
            f"Eigenfaces matrix shape is incorrect. Expected {3, 2} got {projections.shape}",
        )
        self.assertTrue(
            np.allclose(projections, test_ef.projections, rtol=0.01),
            msg="The projection is incorrect",
        )
        success_msg = "Projection"
        print_success_message(success_msg)


class TestAudioCompression(unittest.TestCase):
    """
    Tests for Audio Compression
    """

    def test_svd_mono(self):
        """
        Test correct implementation of SVD calculation for single channel audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        U, S, V = ac.svd(test_ac.mono_audio)
        self.assertEqual(np.allclose(U, test_ac.Ug), True, "U is incorrect")
        self.assertEqual(np.allclose(S, test_ac.Sg), True, "S is incorrect")
        self.assertEqual(np.allclose(V, test_ac.Vg), True, "V is incorrect")
        success_msg = "SVD calculation - single channel audio"
        print_success_message(success_msg)

    def test_compress_mono(self):
        """
        Test correct implementation of audio compression for single channel audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        U, S, V = test_ac.Ug, test_ac.Sg, test_ac.Vg
        Uc, Sc, Vc = ac.compress(U, S, V, 2)
        self.assertEqual(
            np.allclose(Uc, test_ac.Ugc), True, "U compression is incorrect"
        )
        self.assertEqual(
            np.allclose(Sc, test_ac.Sgc), True, "S compression is incorrect"
        )
        self.assertEqual(
            np.allclose(Vc, test_ac.Vgc), True, "V compression is incorrect"
        )
        success_msg = "Audio compression - single channel audio"
        print_success_message(success_msg)

    def test_rebuild_svd_mono(self):
        """
        Test correct implementation of SVD reconstruction for single channel audios
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        Xrebuild_g = ac.rebuild_svd(test_ac.Ugc, test_ac.Sgc, test_ac.Vgc)
        self.assertEqual(
            np.allclose(Xrebuild_g, test_ac.Xrebuild_g),
            True,
            "Reconstruction is incorrect",
        )
        success_msg = "SVD reconstruction - single channel audio"
        print_success_message(success_msg)

    def test_compression_ratio_mono(self):
        """
        Test correct implementation of compression ratio calculation for single channel audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        cr = ac.compression_ratio(test_ac.mono_audio, 2)
        self.assertEqual(
            np.allclose(cr, test_ac.cr_g), True, "Compression ratio is incorrect"
        )
        success_msg = "Compression ratio - single channel audio"
        print_success_message(success_msg)

    def test_recovered_variance_proportion_mono(self):
        """
        Test correct implementation of recovered variance proportion calculation for single channel audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        rvp = ac.recovered_variance_proportion(test_ac.Sg, 2)
        self.assertEqual(
            np.allclose(rvp, test_ac.rvp_g),
            True,
            "Recovered variance proportion is incorrect",
        )
        success_msg = "Recovered variance proportion - single channel audio"
        print_success_message(success_msg)

    def test_svd_stereo(self):
        """
        Test correct implementation of SVD calculation for stereo audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        U, S, V = ac.svd(test_ac.stereo_audio)
        self.assertEqual(np.allclose(U, test_ac.Uc), True, "U is incorrect")
        self.assertEqual(np.allclose(S, test_ac.Sc), True, "S is incorrect")
        self.assertEqual(np.allclose(V, test_ac.Vc), True, "V is incorrect")
        success_msg = "SVD calculation - stereo audio"
        print_success_message(success_msg)

    def test_compress_stereo(self):
        """
        Test correct implementation of audio compression for stereo audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        U, S, V = test_ac.Uc, test_ac.Sc, test_ac.Vc
        Uc, Sc, Vc = ac.compress(U, S, V, 2)
        self.assertEqual(
            np.allclose(Uc, test_ac.Ucc), True, "U compression is incorrect"
        )
        self.assertEqual(
            np.allclose(Sc, test_ac.Scc), True, "S compression is incorrect"
        )
        self.assertEqual(
            np.allclose(Vc, test_ac.Vcc), True, "V compression is incorrect"
        )
        success_msg = "Audio compression - stereo audio"
        print_success_message(success_msg)

    def test_rebuild_svd_stereo(self):
        """
        Test correct implementation of SVD reconstruction for stereo audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        Xrebuild_c = ac.rebuild_svd(test_ac.Ucc, test_ac.Scc, test_ac.Vcc)
        self.assertEqual(
            np.allclose(Xrebuild_c, test_ac.Xrebuild_c),
            True,
            "Reconstruction is incorrect",
        )
        success_msg = "SVD reconstruction - stereo audio"
        print_success_message(success_msg)

    def test_compression_ratio_stereo(self):
        """
        Test correct implementation of compression ratio calculation for stereo audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        cr = ac.compression_ratio(test_ac.stereo_audio, 2)
        self.assertEqual(
            np.allclose(cr, test_ac.cr_c), True, "Compression ratio is incorrect"
        )
        success_msg = "Compression ratio - stereo audio"
        print_success_message(success_msg)

    def test_recovered_variance_proportion_stereo(self):
        """
        Test correct implementation of recovered variance proportion calculation for stereo audio
        """
        ac = AudioCompression()
        test_ac = AC_Test()
        rvp = ac.recovered_variance_proportion(test_ac.Sc, 2)
        self.assertEqual(
            np.allclose(rvp, test_ac.rvp_c),
            True,
            "Recovered variance proportion is incorrect",
        )
        success_msg = "Recovered variance proportion - stereo audio"
        print_success_message(success_msg)


class TestSVDRecommender(unittest.TestCase):
    """
    Tests for SVD Recommender
    """

    def test_recommender_svd(self):
        """
        Test
        """
        recommender = SVDRecommender()
        test_recommender = SVDRecommender_Test()
        R, _, _ = recommender.create_ratings_matrix(test_recommender.ratings_df)
        U_k, V_k = recommender.recommender_svd(R, 10)
        my_slice_U_k, my_slice_V_k = test_recommender.get_slice_UV(U_k, V_k)
        correct_slice_U_k, correct_slice_V_k = (
            test_recommender.slice_U_k,
            test_recommender.slice_V_k,
        )
        self.assertTrue(
            np.all(U_k.shape == test_recommender.U_k_expected_shape),
            "recommender_svd() function returning incorrect U_k shape",
        )
        self.assertTrue(
            np.all(V_k.shape == test_recommender.V_k_expected_shape),
            "recommender_svd() function returning incorrect V_k shape",
        )
        self.assertEqual(
            np.allclose(my_slice_U_k, correct_slice_U_k),
            True,
            "recommender_svd() function returning incorrect U_k",
        )
        self.assertEqual(
            np.allclose(my_slice_V_k, correct_slice_V_k),
            True,
            "recommender_svd() function returning incorrect V_k",
        )
        success_msg = "recommender_svd() function"
        print_success_message(success_msg)

    def test_predict(self):
        """
        Test
        """
        recommender = SVDRecommender()
        recommender.load_movie_data()
        test_recommender = SVDRecommender_Test()
        R, users_index, movies_index = recommender.create_ratings_matrix(
            test_recommender.complete_ratings_df
        )
        mask = np.isnan(R)
        masked_array = np.ma.masked_array(R, mask)
        r_means = np.array(np.mean(masked_array, axis=0))
        R_filled = masked_array.filled(r_means)
        R_filled = R_filled - r_means
        U_k, V_k = recommender.recommender_svd(R_filled, k=8)
        movie_recommendations = recommender.predict(
            R,
            U_k,
            V_k,
            users_index,
            movies_index,
            test_recommender.test_user_id,
            test_recommender.movies_pool,
        )
        print("Top 3 Movies the User would want to watch:")
        for movie in movie_recommendations:
            print(movie)
        print("--------------------------------------------------------------")
        self.assertEqual(
            len(movie_recommendations)
            == len(test_recommender.predict_expected_outputs),
            True,
            "predict() function is not returning the correct number of recommendations",
        )
        self.assertEqual(
            (movie_recommendations == test_recommender.predict_expected_outputs).all(),
            True,
            "predict() function is not returning the correct recommendations",
        )
        success_msg = "predict() function"
        print_success_message(success_msg)


class TestRegression(unittest.TestCase):
    """
    Tests for Regression
    """

    def test_rmse(self):
        """
        Test correct implementation of linear regression rmse
        """
        reg = Regression()
        test_reg = Regression_Test()
        rmse_test = np.allclose(
            reg.rmse(test_reg.predict, test_reg.y_all), test_reg.rmse
        )
        self.assertTrue(rmse_test, "RMSE is incorrect")
        success_msg = "RMSE"
        print_success_message(success_msg)

    def test_construct_polynomial_feats(self):
        """
        Test correct implementation of polynomial feature construction
        """
        reg = Regression()
        test_reg = Regression_Test()
        poly_feat_test = np.allclose(
            reg.construct_polynomial_feats(test_reg.x_all, 2), test_reg.construct_poly
        )
        self.assertTrue(poly_feat_test, "Polynomial features are incorrect")
        success_msg = "Polynomial feature construction"
        print_success_message(success_msg)

    def test_predict(self):
        """
        Test correct implementation of linear regression prediction
        """
        reg = Regression()
        test_reg = Regression_Test()
        predict_test = np.allclose(
            reg.predict(test_reg.x_all_feat, test_reg.true_weight), test_reg.predict
        )
        self.assertTrue(predict_test, "Prediction is incorrect")
        success_msg = "Linear regression prediction"
        print_success_message(success_msg)

    def test_linear_fit_closed(self):
        """
        Test correct implementation of closed form linear regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        linear_closed_test = np.allclose(
            reg.linear_fit_closed(test_reg.x_all_feat, test_reg.y_all),
            test_reg.linear_closed,
            rtol=0.0001,
        )
        self.assertTrue(linear_closed_test, "Weights are incorrect")
        success_msg = "Closed form linear regression"
        print_success_message(success_msg)

    def test_linear_fit_GD(self):
        """
        Test correct implementation of gradient descent linear regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        linear_GD, linear_GD_loss = reg.linear_fit_GD(
            test_reg.x_all_feat, test_reg.y_all
        )
        lgd_test = np.allclose(linear_GD, test_reg.linear_GD)
        lgd_loss_test = np.allclose(linear_GD_loss, test_reg.linear_GD_loss)
        self.assertTrue(lgd_test, "Weights are incorrect")
        self.assertTrue(lgd_loss_test, "Loss is incorrect")
        success_msg = "Gradient descent linear regression"
        print_success_message(success_msg)

    def test_linear_fit_SGD(self):
        """
        Test correct implementation of stochastic gradient descent linear regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        linear_SGD, linear_SGD_loss = reg.linear_fit_SGD(
            test_reg.x_all_feat, test_reg.y_all, 1
        )
        lsgd_test = np.allclose(linear_SGD, test_reg.linear_SGD)
        lsgd_loss_test = np.allclose(linear_SGD_loss, test_reg.linear_SGD_loss)
        self.assertTrue(lsgd_test, "Weights are incorrect")
        self.assertTrue(lsgd_loss_test, "Loss is incorrect")
        success_msg = "Stochastic gradient descent linear regression"
        print_success_message(success_msg)

    def test_linear_fit_MBGD(self):
        """
        Test correct implementation of mini-batch gradient descent linear regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        linear_MBGD, linear_MBGD_loss = reg.linear_fit_MBGD(
            test_reg.x_all_feat, test_reg.y_all, 5, 10
        )
        lmbgd_test = np.allclose(linear_MBGD, test_reg.linear_MBGD)
        lmbgd_loss_test = np.allclose(linear_MBGD_loss, test_reg.linear_MBGD_loss)
        self.assertTrue(lmbgd_test, "Weights are incorrect")
        self.assertTrue(lmbgd_loss_test, "Loss is incorrect")
        success_msg = "Mini-batch gradient descent linear regression"
        print_success_message(success_msg)

    def test_ridge_fit_closed(self):
        """
        Test correct implementation of closed form ridge regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        ridge_closed_test = np.allclose(
            reg.ridge_fit_closed(test_reg.x_all_feat, test_reg.y_all, 10),
            test_reg.ridge_closed,
        )
        self.assertTrue(ridge_closed_test, "Weights are incorrect")
        success_msg = "Closed form ridge regression"
        print_success_message(success_msg)

    def test_ridge_fit_GD(self):
        """
        Test correct implementation of gradient descent ridge regression
        """
        error_atolerance = 1e-10
        reg = Regression()
        test_reg = Regression_Test()
        ridge_GD, ridge_GD_loss = reg.ridge_fit_GD(
            test_reg.x_all_feat, test_reg.y_all, 50000, 10
        )
        rgd_test = np.allclose(ridge_GD, test_reg.ridge_GD, atol=error_atolerance)
        rgd_loss_test = np.allclose(
            ridge_GD_loss, test_reg.ridge_GD_loss, atol=error_atolerance
        )
        rsgd_bias_incorrect = np.allclose(
            ridge_GD, test_reg.ridge_GD_bias_incorrect, atol=error_atolerance
        )
        self.assertFalse(
            rsgd_bias_incorrect,
            "Weights are incorrect. Make sure that you handle the bias term correctly.",
        )
        self.assertTrue(rgd_test, "Weights are incorrect")
        self.assertTrue(rgd_loss_test, "Loss is incorrect")
        success_msg = "Gradient descent ridge regression"
        print_success_message(success_msg)

    def test_ridge_fit_SGD(self):
        """
        Test correct implementation of stochastic gradient descent ridge regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        ridge_SGD, ridge_SGD_loss = reg.ridge_fit_SGD(
            test_reg.x_all_feat, test_reg.y_all, 20, 1
        )
        rsgd_test = np.allclose(ridge_SGD, test_reg.ridge_SGD)
        rsgd_loss_test = np.allclose(ridge_SGD_loss, test_reg.ridge_SGD_loss)
        rsgd_bias_incorrect = np.allclose(ridge_SGD, test_reg.ridge_SGD_bias_incorrect)
        self.assertFalse(
            rsgd_bias_incorrect,
            "Weights are incorrect. Make sure that you handle the bias term correctly.",
        )
        self.assertTrue(rsgd_test, "Weights are incorrect")
        self.assertTrue(rsgd_loss_test, "Loss is incorrect")
        success_msg = "Stochastic gradient descent ridge regression"
        print_success_message(success_msg)

    def test_ridge_fit_MBGD(self):
        """
        Test correct implementation of mini-batch gradient descent ridge regression
        """
        reg = Regression()
        test_reg = Regression_Test()
        ridge_MBGD, ridge_MBGD_loss = reg.ridge_fit_MBGD(
            test_reg.x_all_feat, test_reg.y_all, 20, 5, 10
        )
        rmbgd_test = np.allclose(ridge_MBGD, test_reg.ridge_MBGD)
        rmbgd_loss_test = np.allclose(ridge_MBGD_loss, test_reg.ridge_MBGD_loss)
        rmbgd_bias_incorrect = np.allclose(
            ridge_MBGD, test_reg.ridge_MBGD_bias_incorrect
        )
        self.assertFalse(
            rmbgd_bias_incorrect,
            "Weights are incorrect. Make sure that you handle the bias term correctly.",
        )
        self.assertTrue(rmbgd_test, "Weights are incorrect")
        self.assertTrue(rmbgd_loss_test, "Loss is incorrect")
        success_msg = "Mini-batch gradient descent ridge regression"
        print_success_message(success_msg)

    def test_ridge_cross_validation(self):
        """
        Test correct implementation of ridge regression cross validation
        """
        reg = Regression()
        test_reg = Regression_Test()
        ridge_cv_test = np.allclose(
            reg.ridge_cross_validation(test_reg.x_all_feat, test_reg.y_all, 3),
            test_reg.cross_val,
        )
        self.assertTrue(ridge_cv_test, "Weights are incorrect")
        success_msg = "Ridge regression cross validation"
        print_success_message(success_msg)


class TestLogisticRegression(unittest.TestCase):
    """
    Tests for Logistic Regression
    """

    def test_sigmoid(self):
        """
        Test correct implementation of sigmoid
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.sigmoid(test_lr.s)
        result_slice = result[:4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.sigmoid_result_slice), "sigmoid incorrect"
        )
        success_msg = "Logistic Regression sigmoid"
        print_success_message(success_msg)

    def test_sigmoid(self):
        """
        Test correct implementation of sigmoid
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.sigmoid(test_lr.s)
        self.assertTrue(
            result.shape == test_lr.s.shape, "sigmoid incorrect: check shape"
        )
        result_slice = result[:4, :4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.sigmoid_result_slice), "sigmoid incorrect"
        )
        success_msg = "Logistic Regression sigmoid"
        print_success_message(success_msg)

    def test_bias_augment(self):
        """
        Test correct implementation of bias_augment
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.bias_augment(test_lr.x)
        result_slice_sum = np.sum(result[:4, :4])
        self.assertTrue(
            np.allclose(result_slice_sum, test_lr.bias_augment_slice_sum),
            "bias_augment incorrect",
        )
        success_msg = "Logistic Regression bias_augment"
        print_success_message(success_msg)

    def test_predict_probs(self):
        """
        Test correct implementation of predict_probs
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.predict_probs(test_lr.x_aug, test_lr.theta)
        self.assertTrue(result.ndim == 2, "predict_probs incorrect: check shape")
        self.assertTrue(
            result.shape[0] == test_lr.x_aug.shape[0],
            "predict_probs incorrect: check shape",
        )
        result_slice = result[:4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.predict_probs_result_slice),
            "predict_probs incorrect",
        )
        success_msg = "Logistic Regression predict_probs"
        print_success_message(success_msg)

    def test_predict_labels(self):
        """
        Test correct implementation of predict_labels
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.predict_labels(test_lr.h_x, test_lr.threshold)
        self.assertTrue(result.ndim == 2, "predict_labels incorrect: check shape")
        self.assertTrue(
            result.shape[0] == test_lr.h_x.shape[0],
            "predict_labels incorrect: check shape",
        )
        result_slice = result[:4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.predict_labels_result_slice),
            "predict_labels incorrect",
        )
        success_msg = "Logistic Regression predict_labels"
        print_success_message(success_msg)

    def test_loss(self):
        """
        Test correct implementation of loss
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.loss(test_lr.y, test_lr.h_x)
        self.assertAlmostEqual(result, test_lr.loss_result, msg="loss incorrect")
        success_msg = "Logistic Regression loss"
        print_success_message(success_msg)

    def test_gradient(self):
        """
        Test correct implementation of gradient
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.gradient(test_lr.x_aug, test_lr.y, test_lr.h_x)
        self.assertTrue(result.ndim == 2, "gradient incorrect: check shape")
        self.assertTrue(
            result.shape[0] == test_lr.x_aug.shape[1], "gradient incorrect: check shape"
        )
        result_slice = result[:4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.gradient_result_slice),
            "gradient incorrect",
        )
        success_msg = "Logistic Regression gradient"
        print_success_message(success_msg)

    def test_accuracy(self):
        """
        Test correct implementation of accuracy
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.accuracy(test_lr.y, test_lr.y_hat)
        self.assertAlmostEqual(result, test_lr.accuracy_result, "accuracy incorrect")
        success_msg = "Logistic Regression accuracy"
        print_success_message(success_msg)

    def test_evaluate(self):
        """
        Test correct implementation of evaluate
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.evaluate(test_lr.x, test_lr.y, test_lr.theta, test_lr.threshold)
        self.assertAlmostEqual(
            result[0], test_lr.evaluate_result[0], msg="evaluate incorrect"
        )
        self.assertAlmostEqual(
            result[1], test_lr.evaluate_result[1], msg="evaluate incorrect"
        )
        success_msg = "Logistic Regression evaluate"
        print_success_message(success_msg)

    def test_fit(self):
        """
        Test correct implementation of fit
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        result = lr.fit(
            test_lr.x,
            test_lr.y,
            test_lr.x,
            test_lr.y,
            test_lr.lr,
            test_lr.epochs,
            test_lr.threshold,
        )
        self.assertTrue(result.ndim == 2, "fit incorrect: check shape")
        self.assertTrue(
            result.shape[0] == test_lr.theta.shape[0], "fit incorrect: check shape"
        )
        result_slice = result[:4]
        self.assertTrue(
            np.allclose(result_slice, test_lr.fit_result_slice), "fit incorrect"
        )
        success_msg = "Logistic Regression fit"
        print_success_message(success_msg)

    def test_thresholding(self):
        """
        Test correct implementation of hyperparameter tuning
        """
        lr = LogisticRegression()
        test_lr = LogisticRegression_Test()
        thresholds = np.array([0.1, 0.3, 0.7, 0.9])
        theta = lr.fit(
            test_lr.x,
            test_lr.y,
            test_lr.x,
            test_lr.y,
            test_lr.lr,
            test_lr.epochs,
            test_lr.threshold,
        )
        result = hyperparameter_tuning(lr, test_lr.x, test_lr.y, theta, thresholds)
        self.assertTrue(result[0] == test_lr.threshold_result[0])
        self.assertAlmostEqual(
            result[1], test_lr.threshold_result[1], msg="accuracies incorrect"
        )
        success_msg = "Hyperparameter Tuning"
        print_success_message(success_msg)


class TestFeatureReduction(unittest.TestCase):
    """
    Tests for Feature Reduction
    """

    def test_forward_selection(self):
        fr = FeatureReduction()
        test_fr = FeatureReduction_Test()
        student_features = fr.forward_selection(
            test_fr.X, test_fr.y, test_fr.significance_level
        )
        correct_features = test_fr.correct_forward
        val = set(student_features) == set(correct_features)
        error_msg = """Your forward selection function did not yield the correct features.
        Common issues with this function are:
        (1) Not using the given regression function
        (2) Not adding a bias term
        (3) Not using the significance level properly
        (4) The order in which the features are removed is incorrect"""
        self.assertTrue(val, error_msg)
        success_msg = "Forward Selection"
        print_success_message(success_msg)

    def test_backward_elimination(self):
        fr = FeatureReduction()
        test_fr = FeatureReduction_Test()
        student_features = fr.backward_elimination(
            test_fr.X, test_fr.y, test_fr.significance_level
        )
        correct_features = test_fr.correct_backward
        val = set(student_features) == set(correct_features)
        error_msg = """Your backward elimination function did not yield the correct features.
        Common issues with this function are:
        (1) Not using the given regression function
        (2) Not adding a bias term
        (3) Not using the significance level properly
        (4) The order in which the features are removed is incorrect"""
        self.assertTrue(val, error_msg)
        success_msg = "Backward Elimination"
        print_success_message(success_msg)


class TestSMOTE(unittest.TestCase):

    def test_simple_confusion_matrix(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        true, pred = sm_data.simple_classification_vectors
        student_conf = sm.generate_confusion_matrix(y_true=true, y_pred=pred)
        correct_conf = sm_data.simple_conf_matrix
        self.assertTrue(
            student_conf.shape == correct_conf.shape,
            f"The confusion matrix should be a square matrix (u, u), where u is the number of unique labels present. Expected {correct_conf.shape}, got {student_conf.shape}",
        )
        self.assertTrue(
            isinstance(student_conf[0, 0], (int, np.integer)),
            f"The entries of a confusion matrix should be integers. Expected int-like, got {type(student_conf[0, 0])}. Try setting the dtype of your numpy array.",
        )
        self.assertFalse(
            np.array_equal(student_conf, correct_conf.T),
            'You have the axes inverted conceptually. Make sure to put "true" and "predicted" on the left and top respectively.',
        )
        self.assertTrue(
            np.array_equal(student_conf, correct_conf),
            "Confusion matrix incorrectly calculated.",
        )
        print_success_message("simple confusion matrix")

    def test_complex_confusion_matrix(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        true, pred = sm_data.multiclass_classification_vectors
        student_conf = sm.generate_confusion_matrix(y_true=true, y_pred=pred)
        correct_conf = sm_data.multiclass_conf_matrix
        self.assertFalse(
            student_conf.shape == (2, 2),
            "In section 6.1, you should design your functions for a multiclass classification result. Your function might only be supporting a binary classification.",
        )
        self.assertTrue(
            student_conf.shape == correct_conf.shape,
            f"The confusion matrix should be a square matrix (u, u), where u is the number of unique labels present in either vector. Expected (4,4), got {student_conf.shape}.",
        )
        self.assertTrue(
            isinstance(student_conf[0, 0], (int, np.integer)),
            f"The entries of a confusion matrix should be integers. Expected int-like, got {type(student_conf[0, 0])}. Try setting the dtype of your numpy array.",
        )
        self.assertFalse(
            np.array_equal(student_conf, correct_conf.T),
            'You have the axes inverted conceptually. Make sure to put "true" and "predicted" on the left and top respectively.',
        )
        self.assertTrue(
            np.array_equal(student_conf, correct_conf),
            "Confusion matrix incorrectly calculated.",
        )
        print_success_message("multiclass confusion matrix")

    def test_interpolate(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        start, end = sm_data.interpolation_vectors
        coeff = sm_data.inter_coeff
        student_point = np.array(sm.interpolate(start, end, coeff))
        correct_point = np.array(sm_data.correct_point)
        flipped_point = np.array(sm_data.flipped_coeff)
        self.assertTrue(
            student_point.shape == correct_point.shape,
            f"The interpolated point should belong to the same dimension space as the input points. Expected {correct_point.shape}, got {student_point.shape}.",
        )
        self.assertTrue(
            isinstance(student_point[0], (float, np.floating)),
            f"The interpolated point should be an array of floats. Expected float-like, got {type(student_point[0])}. Try setting the dtype of your numpy array.",
        )
        self.assertFalse(
            np.allclose(student_point, flipped_point),
            "You flipped the start and the end, or you did the interpolation calculation backwards. You're close!",
        )
        self.assertTrue(
            np.allclose(student_point, correct_point),
            f"""Your calculation of the interpolation was incorrect. Expected
{correct_point}
 got
{student_point}""",
        )
        print_success_message("interpolation")

    def test_knn(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        points = sm_data.knn_points
        student_nns = sm.k_nearest_neighbors(points, 3)
        correct_nns = sm_data.three_nn
        self.assertTrue(
            student_nns.shape == correct_nns.shape,
            f"Each point should map to its k nearest neighbors, in the shape (N, k). Expected {correct_nns.shape}, got {student_nns.shape}.",
        )
        self.assertTrue(
            isinstance(student_nns[0, 0], (int, np.integer)),
            f"The entries of nearest neighbors array should be integer indices. Expected int-like, got {type(student_nns[0, 0])}. Try setting the dtype of your numpy array.",
        )
        is_correct = True
        for i in range(correct_nns.shape[0]):
            if set(correct_nns[i]) != set(student_nns[i]):
                is_correct = False
                break
        self.assertTrue(is_correct, f"One or more of your neighborhoods is incorrect.")
        print_success_message("k nearest neighbors")

    def test_smote(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        X = sm_data.X
        y = sm_data.y
        student_output = sm.smote(X, y, 3, [0, 1])
        self.assertTrue(
            len(student_output) == 2,
            f"You should return an iterable (e.g., Tuple) containing two items: synthetic X and synthetic y. Expected 2, got {len(student_output)}",
        )
        student_synth_X, student_synth_y = student_output
        self.assertTrue(
            student_synth_y.shape == (30,),
            f"Since |maj|=40 and |min|=10, you should have generated 30 points. Expected (30,), got {student_synth_y.shape}.",
        )
        self.assertTrue(
            student_synth_X.shape == (30, 4),
            f"Since |maj|=40 and |min|=10, you should have generated 30 points. Expected (30,4), got {student_synth_X.shape}.",
        )
        self.assertTrue(
            np.all(student_synth_y == 1),
            "The generated y values should be all the minority label: 1",
        )
        X_minority = X[y == 1]
        X_floor = np.min(X_minority, axis=0)
        X_ceil = np.max(X_minority, axis=0)
        is_correct = True
        for i in range(student_synth_X.shape[0]):
            for j in range(student_synth_X.shape[1]):
                if student_synth_X[i, j] < X_floor[j]:
                    is_correct = False
                    break
                if student_synth_X[i, j] > X_ceil[j]:
                    is_correct = False
                    break
        self.assertTrue(
            is_correct,
            f"You are generating points outside of the convex hull of the minority class. Either you are incorrectly applying interpolate or interpolate is incorrect.",
        )
        print_success_message("SMOTE")

    def test_threshold_eval(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        threshold = sm_data.threshold_roc
        y_true = sm_data.y_true_roc
        y_pred = sm_data.y_pred_roc
        fpr_tpr = sm.threshold_eval(y_true, y_pred, threshold)
        self.assertTrue(
            len(fpr_tpr) == 2,
            f"Threshold function should return a tuple of size 2, containing (FPR, TPR). But the function returned {len(fpr_tpr)} elements.",
        )
        self.assertTrue(
            isinstance(fpr_tpr[0], float) and isinstance(fpr_tpr[1], float),
            f"Threshold function should return a tupe of floats, containing (FPR, TPR). But the function returned types ({type(fpr_tpr[0])}) and ({type(fpr_tpr[1])}).",
        )
        self.assertTrue(
            np.isclose(fpr_tpr[0], sm_data.fpr_test),
            f"The False Positive Rate at threshold {threshold} is incorrect. Please check the order and the value of the FPR and TPR in the return tuple.",
        )
        self.assertTrue(
            np.isclose(fpr_tpr[1], sm_data.tpr_test),
            f"The True Positive Rate at threshold {threshold} is incorrect. Please check the order and the value of the FPR and TPR in the return tuple.",
        )
        print_success_message("Threshold evaluation")

    def test_integrate_curve(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        roc_points = sm_data.fpr_tpr_test
        auc_stu = sm.integrate_curve(roc_points)
        auc_exp = sm_data.auc
        self.assertTrue(
            isinstance(auc_stu, float),
            f"The AUC value should be a float, but the function returned is of type {type(auc_stu)}.",
        )
        self.assertTrue(
            np.isclose(auc_stu, auc_exp),
            f"The AUC value is incorrect. Expected {auc_exp}, got {auc_stu}.",
        )
        print_success_message("Integrating ROC Curve to get AUC")

    def test_generate_roc(self):
        sm = SMOTE()
        sm_data = SMOTE_Test()
        y_true = sm_data.y_true_roc
        y_pred = sm_data.y_pred_roc
        roc_points = sm.generate_roc(y_true, y_pred)
        size_roc = len(sm_data.fpr_tpr_test)
        self.assertTrue(
            len(roc_points) == size_roc,
            f"The ROC curve should have {size_roc} points, ensure that all the thresholds values including the edge cases are considered. ",
        )
        self.assertTrue(
            np.array(roc_points).shape[1] == 2,
            f"The ROC Curve should be a 2D array of shape (n, 2), where n is the number of thresholds considered.",
        )
        self.assertTrue(
            np.allclose(roc_points, sm_data.fpr_tpr_test),
            f"The ROC curve points are incorrect. Please check the order and the values of fpr and tpr.",
        )
        print_success_message("Generate ROC Points")
