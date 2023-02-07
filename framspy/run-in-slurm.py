#!/usr/bin/env python3

import sys
import subprocess
import datetime
import re
import time
import itertools
import json
from enum import Enum

class OnError(Enum):
    IGNORE = 1
    TERMINATE = 2
    REPEAT = 3

def load_json(filename):
    with open(filename) as f:
        try:
            task_batch = json.load(f)
        except json.decoder.JSONDecodeError:
            return None, None

    ids = {}
    values = []

    for key in task_batch:
        if key != "command":
            ids[key] = len(values)
            if isinstance(task_batch[key], list):
                values.append(task_batch[key])
            elif isinstance(task_batch[key], str) and re.match("^\(-?[0-9]+:-?[0-9]+\)$", task_batch[key]):
                first = int(task_batch[key][1:].split(":")[0])
                second = int(task_batch[key][:-1].split(":")[1])
                values.append([i for i in range(first, second+1)])
            else:
                values.append([task_batch[key]])

    lines = []
    for p in itertools.product(*values):
        template = task_batch["command"]
        for key in ids:
            template = re.sub(r"%"+str(key)+r"", str(p[ids[key]]), template)
            #template = re.sub(r"%\b"+str(key)+r"\b", str(p[ids[key]]), template)
        lines.append(template)

   # print(lines)
	
    return lines, 1

def load_lines(filename):
    no_of_lines = 0
    commands = []
    with open(filename) as f:
        for line in f:
            if not(re.match("^\s*#", line) or re.match("^\s*$", line)):
                lines = expand_line(line)
                if len(lines) > 0:
                    no_of_lines += 1
                for l in lines:
                    commands.append(l)
    return commands, no_of_lines

def expand_line(line):
    params = []
    line_template = ""
    post = ""

    # test if valid parametrization
    splitted = line.split("{")
    wrong = False
    if len(splitted[0].split("}")) > 1:
        wrong = True
    for s in splitted[1:]:
        if len(s.split("}")) != 2:
            wrong = True
            break
    if wrong:
        print("*** WARNING! Incorrect parameter definition in line \"" + line + "\". The line will be ignored.")
        return []

    # if valid, expand the line based on the cartesian product of the parameters
    while True:
        parts = line.split("{", 1)
        if len(parts) == 1: # if no more parameters
            break
        pre = parts[0]
        parts = parts[1].split("}", 1)
        mid = parts[0]
        post = parts[1]
        params.append(mid.split(","))
        line_template += pre + "{}"
        line = post
    line_template += post
    if len(params) == 0:
        line_template = line

    lines = []
    for p in itertools.product(*params):
        lines.append(line_template.format(*p))
    return lines


def construct_batchfile(command, max_time, partitions, save_out, save_err, additionals):
    lines = []
    lines.append("#!/bin/bash\n")
    lines.append("#SBATCH --time=" + ("48:00:00" if len(max_time) == 0 else max_time) + "\n")
    if len(partitions)>0:
        lines.append("#SBATCH --partition=" + partitions + "\n")
    #if email != "":
    #    lines.append("#SBATCH --mail-type=END\n")
    #    lines.append("#SBATCH --mail-user=" + email + "\n")
	
    #add defaults to the additionals
    try:
        with open("srun_default", "r") as defaults:
            for line in defaults:
                additionals.append(line)
    except IOError:
        pass
	
    for line in additionals:
        lines.append("#SBATCH " + line + "\n")

    lines.append("\n")
    lines.append("srun " + ("" if save_out else "-o /dev/null ") + ("" if save_err else "-e /dev/null ") + command)

    return "".join(lines)


def run_command(cmd, capture_output, on_error):
    repeat = True
    while repeat:
        repeat = False
        output = ""
        err = ""

        if capture_output:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (output, err) = p.communicate()
            output = output.decode("utf-8")
            err = err.decode("utf-8")
        else:
            p = subprocess.Popen(cmd, shell=True)
        p_status = p.wait()

        if p_status != 0:
            print("*** Running a shell command returned non-zero status = %d." % p_status)
            print("*** Command: " + cmd)
            print("*** Output stream: " + str(output))
            print("*** Error stream: " + str(err))
            if on_error == OnError.TERMINATE:
                print("*** The script will now terminate.")
                quit()  # THE WHOLE PROGRAM ENDS HERE
            elif on_error == OnError.REPEAT:
                print("*** Attempting to repeat the command in 5sec...")
                time.sleep(5)
                repeat = True

    return output

def continue_question(text):
    while True:
        confirmation = input(text + " (y/n)? ")
        if confirmation.lower() in ["y", "yes"]:
            return True
        elif confirmation.lower() in ["n", "no"]:
            return False

def main(argv):
    jobs = int(run_command("squeue -a -h | wc -l", True, OnError.TERMINATE))
    myjobs = int(run_command("echo $USER | xargs squeue -h -u | wc -l", True, OnError.TERMINATE))

    print("There are currently " + str(jobs) + " jobs in the slurm queue, " + str(myjobs) + " of which were submitted by this user:")
    print(run_command("echo $USER | xargs squeue -o '%.18i %.10M %30R %.40j   %.2t' -h -u | sed 's/lab-..-../lab-...../g' |  sort -t '~' --key=1.31 | uniq -s 31 -c  |  sort -t '~' --key=1.69", True, OnError.TERMINATE)) #od update slurma w kwietniu 2017, slurm wprowadzil nowy status "(launch failed requeued held)", ktory w przeciwienstwie do wszystkich innych statusow ma w sobie spacje. Dlatego nie mozemy juz polegac na spacji jako separatorze kolumn w wyniku komendy squeue. Musimy niestety wykorzystywac adresowanie kolumn oparte na znakach. Uzywamy '~' jako separator kolumn, ktory zakladamy ze nigdy nie wystepuje w wyniku squeue, wiec caly wiersz jest traktowany jak jedna kolumna ktora mozna adresowac numerami znakow. Ponadto musimy tez podmieniac wszystkie nazwy komputerow w labach na taki sam staly ciag znakow, bo chcemy grupowac po szczegolowych statusach - np. skrocony status PD (pending) moze byc spowodowany przez szczegolowy status (Priority), (Resources) albo (launch failed requeued held) - i chcemy te wszystkie trzy powody widziec osobno, a one sa podawane w tej samej kolumnie co nazwa komputera na ktorej dziala zadanie (a te nazwy z kolei chcemy "sklejac" bo nie sa dla nas wazne).

    if len(argv) == 1:
        print("#" * 76)
        print("### Usage: python3 " + argv[0] + " commands-to-run.txt")
        print("### This script requires one argument: name of the file with commands which will be run in slurm, "
              "one command in each line.")
        print("### Braces {} can be used in commands to introduce sets of parametr values that the braces should be replaced with. "
              "Values of the parameter should be given inside the braces, separated with commas, with no spaces. "
              "Multiple parameters can be used in one command, in which case tasks will be generated based of the cartesian product of these parameters.")
        print("### Each command will be submited to the queue separately through sbatch, "
              "so after the script terminates you will be able to close your terminal safely.")
        print("#" * 76)
        quit()

    filename = argv[1]
    commands, no_of_lines = load_json(filename)
    if not commands:
        commands, no_of_lines = load_lines(filename)

    print('-' * 76)
    print("%d line(s) (%d task(s)) found in the '%s' file. %d job(s) will be created." % (no_of_lines, len(commands), argv[1], len(commands)))

    if not continue_question("Continue"):
        return

    max_time = input("Enter the maximum time (in hours) per job (eg. \"01:15:00\"; Enter = 48h)\n")
    partitions = input("Enter partition constraints (eg. \"lab-43,lab-44\"; Enter = no constraints)\n")

    save_out = input("Save STDOUT of each command in a separate file? (y/n; Enter = yes)\n")
    save_out = (False if save_out.lower() in ["n", "no"] else True)

    save_err = input("Save STDERR of each command in a separate file? (y/n; Enter = yes)\n")
    save_err = (False if save_err.lower() in ["n", "no"] else True)

    #email = input("Address to send emails to when the last job starts/fails/finishes? (eg. john.smith@gmail.com, Enter = do not send email)?\n")

    additionals = []
    while True:
        addit = input("Add any additional sbatch options you want such as '--exclude=lab-al-9' or '--nice' (Enter = no more options)\n")
        if addit == "":
            break
        else:
            additionals.append(addit)


    now = datetime.datetime.now()
    default = str(now.day) + "-" + str(now.hour) + "-" + str(now.minute)
    cname = input("Enter custom name for the jobs; the name will be visible in squeue (Enter = " + default + ")\n")
    keepcharacters = ('_', '-')
    cname = "".join([c for c in cname if c.isalpha() or c.isdigit() or c in keepcharacters]).rstrip()
    cname = cname if cname != "" else default
    filename = "ris-" + cname + ".sh"

    print("A sample sbatch file for the first command from '%s' is shown below:" % argv[1])
    print("#########################")
    print(construct_batchfile(commands[0], max_time, partitions, save_out, save_err, additionals))
    print("#########################")

    if not continue_question("Confirm"):
        return


    i = 1
    for line in commands:
        print("***** Running job " + str(i) + "/"  + str(len(commands)) + " *****")

        run_command("touch " + filename, True, OnError.TERMINATE)

        f = open(filename, "w")
        f.write(construct_batchfile(line, max_time, partitions, save_out, save_err, additionals))
        f.close()

        run_command("sbatch " + filename, False, OnError.TERMINATE)

        run_command("rm " + filename, True, OnError.TERMINATE)

        i += 1
        
        #time.sleep(2)
        #if i%100==0:
        #    time.sleep(5)

if __name__ == "__main__":
    main(sys.argv)
