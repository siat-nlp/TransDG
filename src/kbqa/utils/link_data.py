# -*- coding: utf-8 -*-


class LinkData:
    def __init__(self, category, start, end, mention, comp, value, name, link_feat, gl_pos=None):
        self.gl_pos = gl_pos
        self.category = category
        self.start = start
        self.end = end
        self.mention = mention
        self.comp = comp
        self.value = str(value)
        self.name = name
        self.link_feat = link_feat      # dictionary

    def serialize(self):
        ret_list = []
        for key in ('category', 'start', 'end', 'mention', 'comp', 'value', 'name', 'link_feat'):
            ret_list.append((key, getattr(self, key)))
        if self.gl_pos is not None:
            ret_list = [('gl_pos', self.gl_pos)] + ret_list
        return ret_list

    def display(self):
        ret_str = ''
        if self.gl_pos is not None:
            ret_str += '#%02d ' % self.gl_pos
        ret_str += '%s: [%d, %d) (%s) %s %s ' % (
            self.category, self.start, self.end, self.mention, self.comp, self.value)
        if self.name != '':
            ret_str += '(%s) ' % self.name
        ret_str += str(self.link_feat)
        return ret_str
