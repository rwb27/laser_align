#!/usr/bin/python
# coding=UTF-8

import curses
import locale

locale.setlocale(locale.LC_ALL,"")

def get_ch_gen(scr):
    while True:
        ch=scr.getch() # pauses until a key's hit
        scr.nodelay(1)
        bs=[]
        while ch != -1: # check if more data is available
            if ch > 255: # directly yield arrowkeys etc as ints
                yield ch
            else:
                bs.append(chr(ch))
                ch=scr.getch()
        scr.nodelay(0)
        for ch in ''.join(bs).decode('utf8'):
            yield ch

def doStuff2(stdscr):
    text = list(u"˙ㄚㄞㄢㄦㄗㄧㄛㄟㄣ"*5)
    import random
    random.shuffle(text)

    stdscr.addstr((''.join(reversed(text)) + '\n').encode("utf-8"))
    for ch in get_ch_gen(stdscr):
        if len(text) == 0:
            break
        if ch == text[-1]:
            stdscr.addstr(text.pop().encode("utf-8"))


def doStuff(stdscr):
    getch=get_ch_gen(stdscr).next

    stdscr.addstr(u'Testing... type å, ä, ö or q to quit\n'.encode("utf-8"))
    while True:
        if getch() in u'åäöq':
            break

curses.wrapper(doStuff)