import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

features, _ = make_blobs(n_samples=10,
                         n_features=2,
                         centers=1,
                         random_state=1)

features[0, 0] = 1000
features[0, 1] = 1000

outlier_detector = EllipticEnvelope(contamination=.1)

outlier_detector.fit(features)

print(outlier_detector.predict(features))

# or we can use interquartile range - rozstep cwiartkowy

feature = features[:, 0]

q1, q3 = np.percentile(feature, [25, 75])
iqr = q3 - q1
lower_boundary = q1 - (iqr * 1.5)
upper_boundary = q3 + (iqr * 1.5)
print(np.where((feature > upper_boundary) | (feature < lower_boundary)))


