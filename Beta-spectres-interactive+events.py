import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

# Константы
m_e = 0.511  # масса электрона в МэВ/c²

def load_nuclide_database(database='beta-database.csv'):
    """Загрузка базы данных радионуклидов"""
    try:
        df = pd.read_csv(database)
        print(f"База данных загружена успешно! Найдено {len(df)} радионуклидов")
        return df
    except FileNotFoundError:
        print(f"Ошибка: файл {database} не найден!")
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
            choice = int(input(f"\n Выберите радионуклид (1-{len(df)}): "))
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
    """Генерация событий методом Монте-Карло (оптимизированная версия)"""
    print(f"\n Генерация {num_events} событий...")
    
    import time  # Добавляем импорт времени
    start_time = time.time()  # Засекаем время начала
    
    nuclide = decay_data['nuclide']
    decay_channels = decay_data['decay_channels']
    
    # 1. Находим максимальную энергию
    max_Q = max(Q for Q, _ in decay_channels)
    
    # 2. Создаем детальную сетку для вычисления спектра
    T_detailed = np.linspace(0, max_Q - 0.001, 10000)
    
    # 3. ОДИН РАЗ вычисляем суммарный спектр
    total_spectrum = np.zeros_like(T_detailed)
    
    for Q, prob in decay_channels:
        # Вычисляем спектр для этого канала
        spectrum = spectrum_vs_kinetic_energy_single(Q, T_detailed)
        
        # Вычисляем площадь (интеграл) под спектром
        area = np.trapz(spectrum, T_detailed)
        
        if area > 0:
            # Нормируем спектр на 1 и умножаем на вероятность канала
            spectrum = spectrum / area * prob
        
        # Добавляем в суммарный спектр
        total_spectrum += spectrum
    
    # 4. Находим максимальное значение суммарного спектра
    max_spectrum = np.max(total_spectrum)
    
    print(f"Максимальная энергия: {max_Q:.4f} МэВ")
    print(f"Максимум суммарного спектра: {max_spectrum:.4f}")
    
    # 5. Функция для быстрого получения значения спектра в точке x
    def get_spectrum_value(x):
        """Быстрое получение значения суммарного спектра в точке x"""
        if x <= 0 or x >= max_Q:
            return 0.0
        
        # Находим индекс ближайшей точки слева на сетке T_detailed
        idx = np.searchsorted(T_detailed, x) - 1
        
        # Проверяем границы
        if idx < 0:
            return total_spectrum[0]
        if idx >= len(T_detailed) - 1:
            return total_spectrum[-1]
        
        # Линейная интерполяция между двумя ближайшими точками
        x1, x2 = T_detailed[idx], T_detailed[idx + 1]
        y1, y2 = total_spectrum[idx], total_spectrum[idx + 1]
        
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    # 6. Генерация событий
    events = []
    attempts = 0
    max_attempts = num_events * 1000  # Защита от бесконечного цикла
    
    while len(events) < num_events and attempts < max_attempts:
        attempts += 1
        
        # Генерируем случайную точку
        x = random.uniform(0, max_Q)  # энергия
        y = random.uniform(0, max_spectrum)  # значение спектра
        
        # БЫСТРО получаем значение спектра из предвычисленного суммарного спектра
        spectrum_value = get_spectrum_value(x)
        
        # Проверяем, попадает ли точка под кривую
        if y <= spectrum_value:
            events.append(x)
    
    # 7. Расчет времени и скорости
    total_time = time.time() - start_time  # Общее время выполнения
    speed = len(events) / total_time  # Скорость генерации (событий/сек)
    
    print(f"Время генерации: {total_time:.3f} сек")
    print(f"Скорость генерации: {speed:.0f} событий/сек")
    
    efficiency = len(events) / attempts * 100
    print(f"Сгенерировано {len(events)} событий из {attempts} попыток (эффективность: {efficiency:.2f}%)")
    
    if len(events) < num_events:
        print(f"Предупреждение: удалось сгенерировать только {len(events)} из {num_events} запрошенных событий")
    
    return events

def save_events_to_file(events, nuclide):
    """Сохранение событий в текстовый файл"""
    file_events = f"{nuclide.replace(' ', '_')}_events.txt"
    
    with open(file_events, 'w') as f:
        for event in events:
            f.write(f"{event:.6f}\n")
    
    print(f"События сохранены в файл: {file_events}")
    return file_events

def plot_generated_events(events, decay_data):
    """Визуализация сгенерированных событий"""
    nuclide = decay_data['nuclide']
    
    # Создаем теоретический спектр
    max_Q = max(Q for Q, _ in decay_data['decay_channels'])
    T_range = np.linspace(0, max_Q - 0.001, 1001)
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
    print("\n Проверка нормировки...")

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
    plt.subplot(2, 1, 1)
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
    plt.subplot(2, 1, 2)
    for i, (Q, prob) in enumerate(decay_channels):
        plt.plot(T_range, spectra_T[i], '--', linewidth=1.5, alpha=0.7, 
                 label=f'Q={Q:.3f} МэВ ({prob:.1%})')
    plt.plot(T_range, total_spectrum_T, 'k-', linewidth=3, label='Суммарный спектр')
    plt.xlabel('Кинетическая энергия Tₑ (МэВ)', fontsize=12)
    plt.ylabel('Нормированный спектр dλ/dTₑ', fontsize=12)
    plt.title(f'Спектр бета-распада {nuclide}\nЗависимость от кинетической энергии')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Спрашиваем пользователя о сохранении
    print("\n" + "="*60)
    save_choice = input("Сохранить график и данные спектра? (y/n): ").strip().lower()
    
    if save_choice == 'y':
        # Создаем имя файла без пробелов и специальных символов
        safe_nuclide = nuclide.replace(' ', '_').replace('-', '_')

        # 1. Сохраняем график как PDF
        pdf_filename = f"Beta_spectrum_of_{safe_nuclide}.pdf"
        
        # Создаем новый рисунок для сохранения
        plt.figure(figsize=(14, 10))
        
            # График 1: Зависимость от импульса
        plt.subplot(2, 1, 1)
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
        plt.subplot(2, 1, 2)
        for i, (Q, prob) in enumerate(decay_channels):
            plt.plot(T_range, spectra_T[i], '--', linewidth=1.5, alpha=0.7, 
                    label=f'Q={Q:.3f} МэВ ({prob:.1%})')
        plt.plot(T_range, total_spectrum_T, 'k-', linewidth=3, label='Суммарный спектр')
        plt.xlabel('Кинетическая энергия Tₑ (МэВ)', fontsize=12)
        plt.ylabel('Нормированный спектр dλ/dTₑ', fontsize=12)
        plt.title(f'Спектр бета-распада {nuclide}\nЗависимость от кинетической энергии')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()  
        # Сохраняем как PDF
        plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()  # Закрываем рисунок чтобы не висел в памяти 

        # 2. Сохраняем данные импульса в текстовый файл
        momentum_filename = f"{safe_nuclide}_momentum_spectrum.txt"
        with open(momentum_filename, 'w') as f:
            # Заголовок с информацией о нуклиде
            f.write(f"# Спектр бета-распада {nuclide}\n")
            f.write(f"# Зависимость от импульса\n")
            f.write(f"# Период полураспада: {half_life}\n")
            f.write(f"# Количество каналов: {len(decay_channels)}\n")
            
            # Заголовки колонок
            header = "Импульс[МэВ/c]"
            for i, (Q, prob) in enumerate(decay_channels):
                header += f"\tКанал_{i+1}_Q={Q:.3f}_prob={prob:.3f}"
            header += "\tСуммарный_спектр"
            f.write(header + "\n")
            
            # Данные
            for j in range(len(p_range)):
                line = f"{p_range[j]:.6f}"
                # Спектры для каждого канала
                for i in range(len(decay_channels)):
                    line += f"\t{spectra_p[i][j]:.6f}"
                # Суммарный спектр
                line += f"\t{total_spectrum_p[j]:.6f}"
                f.write(line + "\n")
        
        # 3. Сохраняем данные кинетической энергии в текстовый файл
        energy_filename = f"{safe_nuclide}_energy_spectrum.txt"
        with open(energy_filename, 'w') as f:
            # Заголовок с информацией о нуклиде
            f.write(f"# Спектр бета-распада {nuclide}\n")
            f.write(f"# Зависимость от кинетической энергии\n")
            f.write(f"# Период полураспада: {half_life}\n")
            f.write(f"# Количество каналов: {len(decay_channels)}\n")
            
            # Заголовки колонок
            header = "Кинетическая_энергия[МэВ]"
            for i, (Q, prob) in enumerate(decay_channels):
                header += f"\tКанал_{i+1}_Q={Q:.3f}_prob={prob:.3f}"
            header += "\tСуммарный_спектр"
            f.write(header + "\n")
            
            # Данные
            for j in range(len(T_range)):
                line = f"{T_range[j]:.6f}"
                # Спектры для каждого канала
                for i in range(len(decay_channels)):
                    line += f"\t{spectra_T[i][j]:.6f}"
                # Суммарный спектр
                line += f"\t{total_spectrum_T[j]:.6f}"
                f.write(line + "\n")
        
        print("\n" + "="*60)
        print("СОХРАНЕНО:")
        print(f"1. График: {pdf_filename}")
        print(f"2. Данные импульса: {momentum_filename}")
        print(f"3. Данные энергии: {energy_filename}")
        
    else:
        print("График и данные не сохранены.")
           
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
    database = 'beta-database.csv'
    df = load_nuclide_database(database)
    
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
                generate_choice = input("\n Хотите сгенерировать события? (y/n): ").lower()
                if generate_choice == 'y':
                    try:
                        num_events = int(input("Сколько событий сгенерировать? "))
                        if num_events > 0:
                            events = generate_events(decay_data, num_events)
                            file_events = save_events_to_file(events, decay_data['nuclide'])
                            
                            # Показываем гистограмму сгенерированных событий (ПРАВИЛЬНАЯ версия)
                            plot_generated_events(events, decay_data)
                        else:
                            print("Число событий должно быть положительным!")
                    except ValueError:
                        print("Пожалуйста, введите целое число!")
            
            # Спрашиваем, продолжить ли
            continue_choice = input("\n Хотите выбрать другой радионуклид? (y/n): ").lower()
            if continue_choice != 'y':
                print("Выход из программы.")
                break