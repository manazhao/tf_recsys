set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'

" The following are examples of different formats supported.
" Keep Plugin commands between vundle#begin/end.
" plugin on GitHub repo
Plugin 'tpope/vim-fugitive'
" plugin from http://vim-scripts.org/vim/scripts.html
" Plugin 'L9'
" Git plugin not hosted on GitHub
Plugin 'git://git.wincent.com/command-t.git'
" git repos on your local machine (i.e. when working on your own plugin)
Plugin 'file:///home/gmarik/path/to/plugin'
" The sparkup vim script is in a subdirectory of this repo called vim.
" Pass the path to set the runtimepath properly.
Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
Plugin 'jlanzarotta/bufexplorer'
Plugin 'Chiel92/vim-autoformat'

" Install L9 and avoid a Naming conflict if you've already installed a
" different version somewhere else.
" Plugin 'ascenator/L9', {'name': 'newL9'}

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this line

set expandtab
set tabstop=2
set shiftwidth=2
set nu
set cc=80
set textwidth=80
" automatically save before leaving a modified buffer.
set autowrite
" display buffer list and prompt to select by number.
nnoremap <F5> :buffers<CR>:buffer<Space>
set wildchar=<Tab> wildmenu wildmode=full
set wildcharm=<C-Z>
nnoremap <F10> :b <C-Z>
set runtimepath^=~/.vim/bundle/ctrlp.vim
set autoread


" key map for switching between windows
" set your own personal modifier key to something handy
let mapleader = "," 

" use ,v to make a new vertical split, ,s for horiz, ,x to close a split
noremap <leader>v <c-w>v<c-w>l
noremap <leader>s <c-w>s<c-w>j
noremap <leader>q <c-w>c<c-w>j
noremap <leader>x :Sex<cr>
noremap <leader>qq <c-w>q
noremap <leader>w :wa<cr>
noremap <leader>ef :!autopep8 --in-place --aggressive --aggressive %<cr>
map <leader>wj <c-w>j
map <leader>wk <c-w>k
map <leader>wl <c-w>l
map <leader>wh <c-w>h
vmap s :sort<cr>
vmap <c-c> "+y

au VimEnter * if &diff | execute 'windo set wrap' | endif
