import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

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
    Qbeta_mev = float(row['Qbeta_mev'])
    
    # Парсим вероятности (разделитель ';')
    try:
        probabilities = [float(x) for x in str(row['probabilities']).split(';')]
    except:
        print(f"Ошибка парсинга вероятностей для {nuclide}")
        return None
    
    # Парсим энергии (разделитель ';')
    try:
        level_energies = [float(x) for x in str(row['level_energies_mev']).split(';')]
    except:
        print(f"Ошибка парсинга энергий для {nuclide}")
        return None
    
    # Проверяем соответствие количеств
    if len(probabilities) != len(level_energies):
        print(f"Ошибка: количество вероятностей ({len(probabilities)}) не совпадает с количеством энергий ({len(level_energies)})")
        return None
    
    # Рассчитаю макс. энергию Q = Qbeta - level_energies
    max_decay_energies = [Qbeta_mev - level_energy for level_energy in level_energies]
    
    # Создаем список каналов распада
    decay_channels = []
    for probability, max_decay_energy in zip(probabilities, max_decay_energies):
        decay_channels.append([max_decay_energy, probability])
    
    # Нормируем вероятности
    total_probability = sum(probability for _, probability in decay_channels)
    if total_probability > 0:
        decay_channels = [[energy, probability/total_probability] for energy, probability in decay_channels]
        print(f"Вероятности нормированы (сумма = {sum(probability for _, probability in decay_channels):.3f})")
    else:
        print("Ошибка: сумма вероятностей равна 0!")
        return None
    
    return {
        'nuclide': nuclide,
        'half_life': half_life,
        'Qbeta_mev': Qbeta_mev,
        'max_decay_energies': max_decay_energies,
        'decay_channels': decay_channels,
        'channels_number': row['levels']
    }

def kinetic_energy_from_momentum(p):
    """Вычисление кинетической энергии из импульса"""
    T_e = np.sqrt(p**2 + m_e**2) - m_e
    return T_e

def spectrum_vs_momentum_single(Q, p):
    """Спектр бета-распада для одного Q-значения как функция импульса"""
    T_e = kinetic_energy_from_momentum(p)
    # Обрезаем спектр при превышении максимальной энергии
    valid = (T_e <= Q) & (T_e > 0)
    spectrum = np.zeros_like(p)
    spectrum[valid] = p[valid]**2 * (Q - T_e[valid])**2
    return spectrum

def spectrum_vs_kinetic_energy_single(Q, T_e):
    """Спектр бета-распада для одного Q-значения как функция кинетической энергии"""
    # Обрезаем спектр при превышении максимальной энергии
    valid = (T_e <= Q) & (T_e > 0)
    spectrum = np.zeros_like(T_e)
    
    p = np.sqrt((T_e[valid] + m_e)**2 - m_e**2)
    dp_dT = (T_e[valid] + m_e) / np.sqrt(T_e[valid]**2 + 2*m_e*T_e[valid])
    spectrum[valid] = p**2 * (Q - T_e[valid])**2 * dp_dT
    
    return spectrum

def generate_events(decay_data, num_events=100):
    """Генерация событий методом Монте-Карло"""
    print(f"\nГенерация {num_events} событий...")
    
    nuclide = decay_data['nuclide']
    decay_channels = decay_data['decay_channels']
    
    # Находим максимальную энергию и максимальное значение спектра
    max_Q = max(Q for Q, _ in decay_channels)
    
    # Создаем детальную сетку для нахождения максимума спектра
    T_detailed = np.linspace(0, max_Q - 0.001, 10000)
    total_spectrum_detailed = np.zeros_like(T_detailed)
    
    # Вычисляем суммарный спектр на детальной сетке
    for Q, prob in decay_channels:
        spectrum_T = spectrum_vs_kinetic_energy_single(Q, T_detailed)
        area_T = np.trapz(spectrum_T, T_detailed)
        if area_T > 0:
            spectrum_T = spectrum_T / area_T * prob
        total_spectrum_detailed += spectrum_T
    
    # Находим максимальное значение спектра
    max_spectrum = np.max(total_spectrum_detailed)
    
    print(f"Максимальная энергия: {max_Q:.4f} МэВ")
    print(f"Максимум спектра: {max_spectrum:.4f}")
    
    # Генерация событий
    events = []
    attempts = 0
    max_attempts = num_events * 1000  # Защита от бесконечного цикла
    
    while len(events) < num_events and attempts < max_attempts:
        attempts += 1
        
        # Генерируем случайную точку
        x = random.uniform(0, max_Q)  # энергия
        y = random.uniform(0, max_spectrum * 1.1)  # значение спектра (с небольшим запасом)
        
        # Вычисляем значение спектра в этой точке
        spectrum_value = 0
        for Q, prob in decay_channels:
            spectrum_T = spectrum_vs_kinetic_energy_single(Q, np.array([x]))
            area_T = np.trapz(spectrum_vs_kinetic_energy_single(Q, T_detailed), T_detailed)
            if area_T > 0:
                spectrum_T = spectrum_T / area_T * prob
            spectrum_value += spectrum_T[0]
        
        # Проверяем, попадает ли точка под кривую
        if y <= spectrum_value:
            events.append(x)
    
    efficiency = len(events) / attempts * 100
    print(f"Сгенерировано {len(events)} событий из {attempts} попыток (эффективность: {efficiency:.2f}%)")
    
    if len(events) < num_events:
        print(f"Предупреждение: удалось сгенерировать только {len(events)} из {num_events} запрошенных событий")
    
    return events

def save_events_to_file(events, nuclide):
    """Сохранение событий в текстовый файл"""
    filename = f"{nuclide.replace(' ', '_')}_events.txt"
    
    with open(filename, 'w') as f:
        for event in events:
            f.write(f"{event:.6f}\n")
    
    print(f"События сохранены в файл: {filename}")
    return filename

def plot_generated_events(events, decay_data):
    """Визуализация сгенерированных событий"""
    nuclide = decay_data['nuclide']
    
    # Создаем теоретический спектр
    max_Q = max(Q for Q, _ in decay_data['decay_channels'])
    T_range = np.linspace(0, max_Q - 0.001, 1000)
    total_spectrum_T = np.zeros_like(T_range)
    
    for Q, prob in decay_data['decay_channels']:
        spectrum_T = spectrum_vs_kinetic_energy_single(Q, T_range)
        area_T = np.trapz(spectrum_T, T_range)
        if area_T > 0:
            spectrum_T = spectrum_T / area_T * prob
        total_spectrum_T += spectrum_T
    
    # нормировка для сравнения с гистограммой
    area_theoretical = np.trapz(total_spectrum_T, T_range)
    if area_theoretical > 0:
        total_spectrum_T_norm = total_spectrum_T / area_theoretical
    else:
        total_spectrum_T_norm = total_spectrum_T
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    
    # Гистограмма сгенерированных событий
    n, bins, patches = plt.hist(events, bins=50, density=True, alpha=0.7, 
                               color='blue', edgecolor='black', label='Сгенерированные события')
    
    # Теоретический спектр (правильно нормированный)
    plt.plot(T_range, total_spectrum_T_norm, 'r-', linewidth=2, label='Теоретический спектр')
    
    plt.xlabel('Кинетическая энергия Tₑ (МэВ)', fontsize=12)
    plt.ylabel('Нормированный спектр dλ/dTₑ', fontsize=12)
    plt.title(f'Сгенерированные события для {nuclide}\n(n={len(events)} событий)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_spectra(decay_data):
    """Построение спектров для выбранного радионуклида"""
    nuclide = decay_data['nuclide']
    half_life = decay_data['half_life']
    decay_channels = decay_data['decay_channels']
    
    # Создаем общий диапазон для импульса и энергии
    max_Q = max(Q for Q, _ in decay_channels)
    p_max_total = np.sqrt((max_Q + m_e)**2 - m_e**2)
    p_range = np.linspace(0, p_max_total, 1001)
    T_range = np.linspace(0, max_Q - 0.001, 1001)
    
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
        area_p = np.trapz(spectrum_p, p_range)  # Площадь под кривой
        if area_p > 0:
            spectrum_p = spectrum_p / area_p * prob
        area_T = np.trapz(spectrum_T, T_range)  # Площадь под кривой  
        if area_T > 0:
            spectrum_T = spectrum_T / area_T * prob
        
        spectra_p.append(spectrum_p)
        spectra_T.append(spectrum_T)
        
        total_spectrum_p += spectrum_p
        total_spectrum_T += spectrum_T

    # После цикла расчета спектров добавим проверку:
    print("\nПроверка нормировки...")

     # Проверка для импульсных спектров
    for i, (Q, prob) in enumerate(decay_channels):
        area_p = np.trapz(spectra_p[i], p_range)
        area_T = np.trapz(spectra_T[i], T_range)
        print(f"Канал {i+1}: Q={Q:.3f} МэВ, prob={prob:.3f}, area_p={area_p:.4f}, area_T={area_T:.4f}")

    # Проверка суммарных спектров
        total_area_p = np.trapz(total_spectrum_p, p_range)
        total_area_T = np.trapz(total_spectrum_T, T_range)
        print(f"Суммарные площади: area_p={total_area_p:.4f}, area_T={total_area_T:.4f}")
    
    # Построение графиков
    plt.figure(figsize=(14, 10))
    
    # График 1: Зависимость от импульса
    plt.subplot(2, 2, 1)
    for i, (Q, prob) in enumerate(decay_channels):
        plt.plot(p_range, spectra_p[i], '--', linewidth=1.5, alpha=0.7, 
                 label=f'Q={Q:.3f} МэВ ({prob:.1%})')
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
                 label=f'Q={Q:.3f} МэВ ({prob:.1%})')
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
                
                # Спрашиваем о генерации событий
                generate_choice = input("\nХотите сгенерировать события? (y/n): ").lower()
                if generate_choice == 'y':
                    try:
                        num_events = int(input("Сколько событий сгенерировать? "))
                        if num_events > 0:
                            events = generate_events(decay_data, num_events)
                            filename = save_events_to_file(events, decay_data['nuclide'])
                            
                            # Показываем гистограмму сгенерированных событий (ПРАВИЛЬНАЯ версия)
                            plot_generated_events(events, decay_data)
                        else:
                            print("Число событий должно быть положительным!")
                    except ValueError:
                        print("Пожалуйста, введите целое число!")
            
            # Спрашиваем, продолжить ли
            continue_choice = input("\nХотите выбрать другой радионуклид? (y/n): ").lower()
            if continue_choice != 'y':
                print("Выход из программы.")
                break