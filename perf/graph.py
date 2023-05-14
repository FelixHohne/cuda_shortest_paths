#!/usr/bin/env python3
import matplotlib.pyplot as plt

def print_row(xs):
    for x in xs:
        print(f' & {x:.3f}', end='')
    print()

def print_table(xs, ys):
    for x in xs:
        print(f' & {x:,}', end='')
    print()
    print_row(ys)

def print_weak(xs, ys):
    print_table(xs, ys)
    fs = [ys[0] / ys[i] for i in range(len(xs))]
    print_row(fs)
    sum = 0
    for f in fs[1:]:
        sum += f
    avg = sum / (len(fs) - 1)
    print(f'Average weak scaling efficiency: {avg:.3f}')

def parse_strength_times(f, ys):
    lines = list(map(lambda x: x.split(' '), [line.strip() for line in f]))
    for line in lines:
        if line[0] == 'Time':
            ys.append(float(line[4]))
    return ys

def plot_weak():
    datasets = ['roadNet-CA', 'soc-LiveJournal', 'soc-pokec-relationships', 'web-Stanford', 'WikiTalk']
    shortnames = {}
    shortnames['roadNet-CA'] = 'roadNet-CA'
    shortnames['soc-LiveJournal'] = 'LiveJournal'
    shortnames['soc-pokec-relationships'] = 'Pokec'
    shortnames['web-Stanford'] = 'webStanford'
    shortnames['WikiTalk'] = 'wikiTalk'

    percentages = [25, 50, 75, 100]
    plt.title('Weak Scaling Performance')
    plt.xlabel('Percentage of Edges Used')
    plt.ylabel('Simulation Time (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    for dataset in datasets:
        xs = percentages[:]
        ys = []
        for percentage in percentages:
            f = open(f'{dataset}-{percentage}-out.txt', 'r')
            ys = parse_strength_times(f, ys)
        color=next(ax._get_lines.prop_cycler)['color']
        plt.plot(xs, ys, label=f'{shortnames[dataset]} Actual', color=color)
        plt.plot(xs, [ys[0] for _ in range(len(xs))], '--', label=f'{shortnames[dataset]} Predicted', color=color)
        print_weak(xs, ys)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig(f'weak', bbox_inches='tight')
    plt.close()

def main():
    plot_weak()
    

if __name__ == '__main__':
    main()
