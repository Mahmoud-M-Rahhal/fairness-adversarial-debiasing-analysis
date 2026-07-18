# Load all necessary packages

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()

from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from sklearn.preprocessing import MaxAbsScaler



# Load dataset and split train/test
dataset_adult = load_preproc_data_adult()
dataset_adult_train, dataset_adult_test = dataset_adult.split([0.6], shuffle=True)
privileged_groups = [{'sex':1}]
unprivileged_groups = [{'sex':0}]

# Scale features
min_max_scaler = MaxAbsScaler()
dataset_adult_train.features = min_max_scaler.fit_transform(dataset_adult_train.features)
dataset_adult_test.features = min_max_scaler.transform(dataset_adult_test.features)

# Train baseline model (without debiasing)
sess = tf.Session()
plain_model = AdversarialDebiasing(privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups, scope_name = "plain_classifier", debias = False, sess = sess)
plain_model.fit(dataset_adult_train)

dataset_plain_test = plain_model.predict(dataset_adult_test)

# Evaluate baseline model
classified_metric_plain_test = ClassificationMetric(dataset = dataset_adult_test, classified_dataset = dataset_plain_test, unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups)
sess.close()
tf.reset_default_graph()
sess = tf.Session()

# Train debiased model
debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups, scope_name = "plain_classifier", debias = True, sess = sess)
debiased_model.fit(dataset_adult_train)

dataset_debiasing_test = debiased_model.predict(dataset_adult_test)


classified_metric_debiasing_test = ClassificationMetric(dataset = dataset_adult_test, classified_dataset = dataset_debiasing_test, unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups)

# Compute fairness and performance metrics
plain_model_equal_opportunity_difference= classified_metric_plain_test.equal_opportunity_difference()
plain_TPR = classified_metric_plain_test.true_positive_rate()
plain_TNR = classified_metric_plain_test.true_negative_rate()
plain_model_classification_accuracy=0.5*(plain_TPR+plain_TNR)
debias_TPR = classified_metric_debiasing_test.true_positive_rate()
debias_TNR = classified_metric_debiasing_test.true_negative_rate()
debias_model_classification_accuracy=0.5*(debias_TPR+debias_TNR)
debias_model_equal_opportunity_difference= classified_metric_debiasing_test.equal_opportunity_difference()

