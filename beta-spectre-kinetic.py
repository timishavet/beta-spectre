import numpy as np
import matplotlib.pyplot as plt

# Константы и параметры
m_e = 0.511  # масса электрона в МэВ/c²
Q = 10.0      # энергия распада в МэВ (можно изменить)

def kinetic_energy_from_momentum(p):
    """Вычисление кинетической энергии из импульса"""
    E_total = np.sqrt(p**2 + m_e**2)  # полная энергия бета частицы (E=sqrt(p^2*m^2))
    T_e = E_total - m_e               # кинетическая энергия бета частицы (T=E-m)
    return T_e

def spectrum_vs_momentum(p):
    """Спектр бета-распада как функция импульса"""
    T_e = kinetic_energy_from_momentum(p)
    # Формула: dλ/dp ~ p²(Q - Tₑ)²
    spectrum = p**2 * (Q - T_e)**2
    # Нормировка на максимальное значение
    spectrum = spectrum / np.max(spectrum)
    return spectrum

def spectrum_vs_kinetic_energy(T_e):
    """Спектр бета-распада как функция кинетической энергии"""
    # Находим соответствующий импульс для каждой энергии p=sqrt((T+m)^2-m^2)
    p = np.sqrt((T_e + m_e)**2 - m_e**2)
    # Формула: dλ/dT = (dλ/dp) * (dp/dT)
    # dp/dT = (T + m_e)/sqrt(T^2+2*m*T) из соотношения p=sqrt((T+m)^2-m^2)
    dp_dT = (T_e + m_e) / np.sqrt(T_e**2+2*m_e*T_e) 
    spectrum = p**2 * (Q - T_e)**2 * dp_dT
    # Нормировка на максимальное значение
    spectrum = spectrum / np.max(spectrum)
    return spectrum

# Создаем диапазоны для импульса и энергии
p_range = np.linspace(0.01, Q, 500)  # импульс в МэВ/c
T_range = np.linspace(0.01, Q - 0.001, 500)  # кинетическая энергия в МэВ

# Вычисляем спектры
spectrum_p = spectrum_vs_momentum(p_range)
spectrum_T = spectrum_vs_kinetic_energy(T_range)

# Построение графиков
plt.figure(figsize=(12, 5))

# График 1: Зависимость от импульса
plt.subplot(1, 2, 1)
plt.plot(p_range, spectrum_p, 'b-', linewidth=2)
plt.xlabel('Импульс p (МэВ/c)', fontsize=12)
plt.ylabel('Нормированный спектр dλ/dp', fontsize=12)
plt.title('Спектр бета-распада: зависимость от импульса')
plt.grid(True, alpha=0.3)

# График 2: Зависимость от кинетической энергии
plt.subplot(1, 2, 2)
plt.plot(T_range, spectrum_T, 'r-', linewidth=2)
plt.xlabel('Кинетическая энергия Tₑ (МэВ)', fontsize=12)
plt.ylabel('Нормированный спектр dλ/dTₑ', fontsize=12)
plt.title('Спектр бета-распада: зависимость от кинетической энергии')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Дополнительно: график в кэВ для более детального просмотра
plt.figure(figsize=(12, 5))

# В кэВ
T_range_keV = np.linspace(10, (Q - 0.001) * 1000, 500)  # в кэВ
spectrum_T_keV = spectrum_vs_kinetic_energy(T_range_keV / 1000)

plt.subplot(1, 2, 1)
plt.plot(T_range_keV, spectrum_T_keV, 'g-', linewidth=2)
plt.xlabel('Кинетическая энергия Tₑ (кэВ)', fontsize=12)
plt.ylabel('Нормированный спектр dλ/dTₑ', fontsize=12)
plt.title('Спектр бета-распада в кэВ')
plt.grid(True, alpha=0.3)

# Сравнение обеих зависимостей на одном графике
plt.subplot(1, 2, 2)
T_e_from_p = kinetic_energy_from_momentum(p_range)
plt.plot(T_e_from_p, spectrum_p, 'b-', label='dλ/dp', linewidth=2)
plt.plot(T_range, spectrum_T, 'r--', label='dλ/dT', linewidth=2)
plt.xlabel('Кинетическая энергия Tₑ (МэВ)', fontsize=12)
plt.ylabel('Нормированный спектр', fontsize=12)
plt.title('Сравнение спектров')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()