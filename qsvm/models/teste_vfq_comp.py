# Exemplo de comparação sistemática
def benchmark_vfq_versions(X_train, y_train, X_test, y_test):
    results = {}

    # Baseline clássico
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    results['Classical SVM'] = svm.score(X_test, y_test)

    # VFQ Original
    vfq_orig = VariationalFullyQuantum(n_qubits=4, n_layers=2)
    vfq_orig.fit(X_train, y_train, epochs=50)
    results['VFQ Original'] = accuracy_score(y_test, vfq_orig.predict(X_test))

    # VFQ com medições múltiplas
    vfq_v1 = VariationalFullyQuantum_V1(measurement_strategy="correlations")
    vfq_v1.fit(X_train, y_train)
    results['VFQ Multi-measurement'] = accuracy_score(y_test, vfq_v1.predict(X_test))

    # VFQ com QNG
    vfq_qng = VariationalFullyQuantum_QNG(optimizer="qng")
    vfq_qng.fit(X_train, y_train)
    results['VFQ QNG'] = accuracy_score(y_test, vfq_qng.predict(X_test))

    # VFQ Ultra
    vfq_ultra = VariationalFullyQuantum_Ultra(
        encoding="amplitude",
        ansatz="strongly_entangling",
        measurement="quantum_kernel"
    )
    vfq_ultra.fit(X_train, y_train)
    results['VFQ Ultra'] = accuracy_score(y_test, vfq_ultra.predict(X_test))

    # Análise de entanglement
    entanglement = vfq_ultra.get_entanglement_measure(X_test[0])
    print(f"Nível de entanglement gerado: {entanglement:.3f}")

    return results