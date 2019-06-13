from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X, y)
