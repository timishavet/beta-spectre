import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Константы
m_e = 0.511  # масса электрона в МэВ/c²

def load_nuclide_database(filename='beta-database.csv'):
    """Загрузка базы данных радионуклидов"""
    try:
        df = pd.read_csv(filename)
        print(f"База данных загружена успешно! Найдено {len(df)} радионуклидов")
        return df
    except FileNotFoundError:
        print(f"Ошибка: файл {filename} не найден!")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return None

def select_nuclide(df):
    """Выбор радионуклида из базы данных"""
    print("\n" + "="*50)
    print("Доступные радионуклиды:")
    for i, (idx, row) in enumerate(df.iterrows()):
        print(f"{i+1:2d}. {row['nuclide']:10} - период полураспада: {row['half_life']}")
    
    while True:
        try:
            choice = int(input(f"\nВыберите радионуклид (1-{len(df)}): "))
            if 1 <= choice <= len(df):
                selected_row = df.iloc[choice-1]
                print(f"\nВыбран: {selected_row['nuclide']}")
                return selected_row
            else:
                print(f"Пожалуйста, введите число от 1 до {len(df)}")
        except ValueError:
            print("Пожалуйста, введите целое число!")

def parse_decay_data(row):
    """Парсинг данных о распаде из строки базы данных"""
    nuclide = row['nuclide']
    half_life = row['half_life']
    
    # Парсим вероятности (разделитель ';')
    try:
        probabilities = [float(x) for x in str(row['probabilities']).split(';')]
    except:
        print(f"Ошибка парсинга вероятностей для {nuclide}")
        return None
    
    # Парсим энергии (разделитель ';')
    try:
        energies = [float(x) for x in str(row['energies_mev']).split(';')]
    except:
        print(f"Ошибка парсинга энергий для {nuclide}")
        return None
    
    # Обрабатываем случай, когда указана только одна энергия (суммарная)
    if len(energies) == 1 and len(probabilities) > 1:
        total_energy = energies[0]
        print(f"Обнаружена суммарная энергия {total_energy} МэВ для {len(probabilities)} каналов")
        print("Энергия будет распределена по каналам: Q_канала = Q_суммарное × вероятность_канала")
        
        # Умножаем суммарную энергию на вероятность для каждого канала
        energies = [total_energy * prob for prob in probabilities]
        
        print("Распределение энергий по каналам:")
        for i, (prob, energy) in enumerate(zip(probabilities, energies)):
            print(f"  Канал {i+1}: вероятность {prob:.3f} -> энергия {energy:.3f} МэВ")
    
    # Проверяем соответствие количеств
    if len(probabilities) != len(energies):
        print(f"Ошибка: количество вероятностей ({len(probabilities)}) не совпадает с количеством энергий ({len(energies)})")
        return None
    
    # Создаем список каналов распада
    decay_channels = []
    for prob, energy in zip(probabilities, energies):
        decay_channels.append([energy, prob])
    
    # Нормируем вероятности
    total_prob = sum(prob for _, prob in decay_channels)
    if total_prob > 0:
        decay_channels = [[energy, prob/total_prob] for energy, prob in decay_channels]
        print(f"Вероятности нормированы (сумма = {sum(prob for _, prob in decay_channels):.3f})")
    else:
        print("Ошибка: сумма вероятностей равна 0!")
        return None
    
    return {
        'nuclide': nuclide,
        'half_life': half_life,
        'decay_channels': decay_channels,
        'decays_number': row['decays_number']
    }

def kinetic_energy_from_momentum(p):
    """Вычисление кинетической энергии из импульса"""
    E_total = np.sqrt(p**2 + m_e**2)
    T_e = E_total - m_e
    return T_e

def spectrum_vs_momentum_single(Q, p):
    """Спектр бета-распада для одного Q-значения как функция импульса"""
    T_e = kinetic_energy_from_momentum(p)
    # Обрезаем спектр при превышении максимальной энергии
    valid = T_e <= Q
    spectrum = np.zeros_like(p)
    spectrum[valid] = p[valid]**2 * (Q - T_e[valid])**2
    return spectrum

def spectrum_vs_kinetic_energy_single(Q, T_e):
    """Спектр бета-распада для одного Q-значения как функция кинетической энергии"""
    # Обрезаем спектр при превышении максимальной энергии
    valid = T_e <= Q
    spectrum = np.zeros_like(T_e)
    
    p = np.sqrt((T_e[valid] + m_e)**2 - m_e**2)
    dp_dT = (T_e[valid] + m_e) / np.sqrt(T_e[valid]**2 + 2*m_e*T_e[valid])
    spectrum[valid] = p**2 * (Q - T_e[valid])**2 * dp_dT
    
    return spectrum

def plot_spectra(decay_data):
    """Построение спектров для выбранного радионуклида"""
    nuclide = decay_data['nuclide']
    half_life = decay_data['half_life']
    decay_channels = decay_data['decay_channels']
    
    # Создаем общий диапазон для импульса и энергии
    max_Q = max(Q for Q, _ in decay_channels)
    p_max_total = np.sqrt((max_Q + m_e)**2 - m_e**2)
    p_range = np.linspace(0.01, p_max_total, 1000)
    T_range = np.linspace(0.01, max_Q - 0.001, 1000)
    
    # Вычисляем суммарные спектры
    total_spectrum_p = np.zeros_like(p_range)
    total_spectrum_T = np.zeros_like(T_range)
    
    # Вычисляем спектры для каждого канала
    spectra_p = []
    spectra_T = []
    
    print("\nВычисление спектров...")
    for i, (Q, prob) in enumerate(decay_channels):
        spectrum_p = spectrum_vs_momentum_single(Q, p_range)
        spectrum_T = spectrum_vs_kinetic_energy_single(Q, T_range)
        
        # Нормируем каждый спектр и умножаем на вероятность
        if np.max(spectrum_p) > 0:
            spectrum_p = spectrum_p / np.max(spectrum_p) * prob
        if np.max(spectrum_T) > 0:
            spectrum_T = spectrum_T / np.max(spectrum_T) * prob
        
        spectra_p.append(spectrum_p)
        spectra_T.append(spectrum_T)
        
        total_spectrum_p += spectrum_p
        total_spectrum_T += spectrum_T
    
    # Построение графиков
    plt.figure(figsize=(14, 10))
    
    # График 1: Зависимость от импульса
    plt.subplot(2, 2, 1)
    for i, (Q, prob) in enumerate(decay_channels):
        plt.plot(p_range, spectra_p[i], '--', linewidth=1.5, alpha=0.7, 
                 label=f'Q={Q} МэВ ({prob:.1%})')
    plt.plot(p_range, total_spectrum_p, 'k-', linewidth=3, label='Суммарный спектр')
    plt.xlabel('Импульс p (МэВ/c)', fontsize=12)
    plt.ylabel('Нормированный спектр dλ/dp', fontsize=12)
    plt.title(f'Спектр бета-распада {nuclide}\nЗависимость от импульса')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 2: Зависимость от кинетической энергии
    plt.subplot(2, 2, 2)
    for i, (Q, prob) in enumerate(decay_channels):
        plt.plot(T_range, spectra_T[i], '--', linewidth=1.5, alpha=0.7, 
                 label=f'Q={Q} МэВ ({prob:.1%})')
    plt.plot(T_range, total_spectrum_T, 'k-', linewidth=3, label='Суммарный спектр')
    plt.xlabel('Кинетическая энергия Tₑ (МэВ)', fontsize=12)
    plt.ylabel('Нормированный спектр dλ/dTₑ', fontsize=12)
    plt.title(f'Спектр бета-распада {nuclide}\nЗависимость от кинетической энергии')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 3: Только суммарные спектры
    plt.subplot(2, 2, 3)
    plt.plot(p_range, total_spectrum_p, 'b-', linewidth=2)
    plt.xlabel('Импульс p (МэВ/c)', fontsize=12)
    plt.ylabel('Нормированный спектр dλ/dp', fontsize=12)
    plt.title(f'Суммарный спектр {nuclide}\nЗависимость от импульса')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(T_range, total_spectrum_T, 'r-', linewidth=2)
    plt.xlabel('Кинетическая энергия Tₑ (МэВ)', fontsize=12)
    plt.ylabel('Нормированный спектр dλ/dTₑ', fontsize=12)
    plt.title(f'Суммарный спектр {nuclide}\nЗависимость от кинетической энергии')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод информации о каналах распада
    print("\n" + "="*60)
    print(f"Параметры распада для {nuclide}:")
    print(f"Период полураспада: {half_life}")
    print(f"Количество каналов распада: {len(decay_channels)}")
    for i, (Q, prob) in enumerate(decay_channels):
        print(f"Канал {i+1}: Q = {Q:.4f} МэВ, вероятность = {prob:.1%}")
    print(f"Максимальная энергия в спектре: {max_Q:.4f} МэВ")
    print("="*60)

# Основная программа
if __name__ == "__main__":
    print("=== Программа построения спектров бета-распада ===")
    print("Загрузка базы данных радионуклидов...")
    
    # Загружаем базу данных (можно указать другой файл)
    filename = 'beta-database.csv'
    df = load_nuclide_database(filename)
    
    if df is not None:
        while True:
            # Выбираем радионуклид
            selected_row = select_nuclide(df)
            
            # Парсим данные о распаде
            decay_data = parse_decay_data(selected_row)
            
            if decay_data is not None:
                # Строим спектры
                plot_spectra(decay_data)
            
            # Спрашиваем, продолжить ли
            continue_choice = input("\nХотите выбрать другой радионуклид? (y/n): ").lower()
            if continue_choice != 'y':
                print("Выход из программы.")
                break