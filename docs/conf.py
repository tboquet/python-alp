# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import alabaster

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinxcontrib.httpdomain',
    'sphinxcontrib.autohttp.flask',
    'alabaster'
]
if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_US'

# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

source_suffix = '.rst'
master_doc = 'index'
project = u'ALP'
year = '2016'
author = u'Thomas Boquet and Paul Lemaître'
copyright = '{0}, {1}'.format(year, author)
version = release = u'0.2.0'

pygments_style = 'sphinx'
templates_path = ['_templates']
extlinks = {
    'issue': ('https://github.com/tboquet/python-alp/issues/%s', '#'),
    'pr': ('https://github.com/tboquet/python-alp/pull/%s', 'PR #'),
}

description = 'Schedule and save your machine learning experiments'
# -- Option for HTML output -----------------------------------------------

html_static_path = ['_static']
html_theme_options = {
    'logo': 'last_bouquetin.svg',
    'logo_name': 'true',
    'description': description,
    'github_button': 'false'
}

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
        'relations.html',
        'last_modified.html'
    ]
}

html_show_sourcelink = True


# Add any paths that contain custom themes here, relative to this directory.

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'alabaster'
    html_theme_path = [alabaster.get_path()]

else:
    from mock import Mock as MagicMock
    import sys

    class Mock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
            return Mock()

    # include the names of your minimal required packages here
    MOCK_MODULES = ['h5py']
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False


html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

keep_warnings = True

add_module_names = False
todo_include_todos = True

mathjax_path = 'https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
