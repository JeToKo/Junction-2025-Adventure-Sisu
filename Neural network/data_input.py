import csv
import datetime
import random

def manipulate_data(day, steps, screen_time, stress, migrane, rng=None):
    if rng is None:
        rng = random

    hour = day.hour

    # Hourly activity pattern for steps (typical low at night, peaks for commute/exercise)
    if 0 <= hour < 6:
        base_steps = rng.gauss(10, 10)
    elif 6 <= hour < 9:  # morning commute/exercise
        base_steps = rng.gauss(600, 200)
    elif 9 <= hour < 12:
        base_steps = rng.gauss(200, 100)
    elif 12 <= hour < 14:  # lunchtime
        base_steps = rng.gauss(300, 150)
    elif 14 <= hour < 17:
        base_steps = rng.gauss(150, 100)
    elif 17 <= hour < 20:  # evening activity
        base_steps = rng.gauss(700, 250)
    else:  # late evening wind-down
        base_steps = rng.gauss(100, 80)

    steps = max(0, int(base_steps + rng.gauss(0, 30)))

    # Screen time in minutes in this hour
    if 0 <= hour < 6:
        base_screen = rng.gauss(5, 5)    # small phone checks
    elif 6 <= hour < 9:
        base_screen = rng.gauss(15, 10)  # quick checks
    elif 9 <= hour < 17:
        base_screen = rng.gauss(25, 15)  # work/school use
    elif 17 <= hour < 22:
        base_screen = rng.gauss(40, 20)  # evening higher usage
    else:
        base_screen = rng.gauss(20, 15)  # late evening

    # If screen time is very high, trigger a migraine with some probability
    if base_screen > 50:
        migrane = 1

    # if high stress from previous hour, increase migraine chance
    if stress > 70:
        migrane = 1

    screen_time = max(0.0, round(base_screen, 1))
    screen_time = min(screen_time, 60.0)

    # Stress: baseline + small contribution from screen time and low sleep hours + migraine spike
    sleep_penalty = 10 if (hour < 6 or hour >= 23) else 0
    stress_noise = rng.gauss(0, 5)
    stress = int(max(0, min(100,
        25 + 0.6 * screen_time + 0.02 * steps + sleep_penalty + (40 if migrane else 0) + stress_noise
    )))

    # Return ISO timestamp for CSV friendliness
    return [day.isoformat(), steps, screen_time, stress, int(bool(migrane))]

def main():
    random.seed(42)
    with open('./phone_data_long', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        day = datetime.datetime(2025, 11, 1, 0, 0)


        while str(day.date()) < '2027-11-1':
            # initial placeholders (values will be overridden by manipulate_data)
            steps = 0
            screen_time = 0.0
            stress = 0
            migrane = 0

            row = manipulate_data(day, steps, screen_time, stress, migrane, rng=random)
            writer.writerow(row)
            print(row)
            day += datetime.timedelta(hours=1)

main()
