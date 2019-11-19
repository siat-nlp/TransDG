# -*- coding: utf-8 -*-
import time


class LogInfo(object):

    lvl = 0
    time_list = []

    active_bw = None
    lvl_cache = 0

    @staticmethod
    def print_func(content):
        if LogInfo.active_bw is not None:
            LogInfo.active_bw.write(content + '\n')
        else:
            print(content)

    @staticmethod
    def redirect(bw, lvl=0):
        if bw is not None and ('w' in bw.mode or 'a' in bw.mode):
            LogInfo.active_bw = bw
            LogInfo.lvl_cache = LogInfo.lvl     # save the level of stdout
            LogInfo.lvl = lvl                   # set the new level when outputting to the new file
        else:
            LogInfo.logs('LogInfo: redirect failed.')

    @staticmethod
    def stop_redirect():
        LogInfo.active_bw = None
        LogInfo.lvl = LogInfo.lvl_cache         # restore the level

    @staticmethod
    def get_blank():
        blank = ''
        for i in range(LogInfo.lvl):
            blank += '  '
        return blank

    @staticmethod
    def begin_track(fmt_string='', *args):
        blank = LogInfo.get_blank()
        if len(args) == 0:
            LogInfo.print_func(blank + '%s' % fmt_string + ' {')
        else:
            fmt = blank + fmt_string + ' {'
            LogInfo.print_func(fmt % args)
        LogInfo.lvl += 1
        LogInfo.time_list.append(time.time())

    @staticmethod
    def logs(fmt_string='', *args):
        blank = LogInfo.get_blank()
        if len(args) == 0:
            LogInfo.print_func(blank + '%s' % fmt_string)
        else:
            fmt = blank + fmt_string
            LogInfo.print_func(fmt % args)

    @staticmethod
    def end_track(fmt_string='', *args):
        if fmt_string != '':
            LogInfo.logs(fmt_string, *args)
        LogInfo.lvl -= 1
        if LogInfo.lvl < 0:
            LogInfo.lvl = 0
        blank = LogInfo.get_blank()
        fmt = blank + '}'
        # if fmt_string != '':
        #     fmt += ' ' + fmt_string
        time_str = ''
        if len(LogInfo.time_list) >= 1:
            elapse = time.time() - LogInfo.time_list.pop()
            time_str = ' [%s]' % LogInfo.show_time(elapse)
        fmt += time_str
        LogInfo.print_func(fmt)

    @staticmethod
    def show_time(elapse):
        ret = ''
        if elapse > 86400:
            d = elapse / 86400
            elapse %= 86400
            ret += '%dd' % d
        if elapse > 3600:
            h = elapse / 3600
            elapse %= 3600
            ret += '%dh' % h
        if elapse > 60:
            m = elapse / 60
            elapse %= 60
            ret += '%dm' % m
        ret += '%.3fs' % elapse
        return ret

    @staticmethod
    def show_line(cnt, num):
        if cnt % num == 0:
            LogInfo.logs("%d lines loaded.", cnt)
