import matplotlib.pyplot as plt

def create_plot(y_test, y_pred, y_pred_poly):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, label='Regresja Liniowa', alpha=0.7)
    plt.scatter(y_test, y_pred_poly, label='Regresja Wielomianowa', alpha=0.7, color='r')
    plt.plot([0, 6], [0, 6], '--', color='gray')  # Idealna linia
    plt.xlabel("Rzeczywista liczba goli")
    plt.ylabel("Przewidywana liczba goli")
    plt.legend()
    plt.title("Porównanie modeli regresji")
    plt.show()