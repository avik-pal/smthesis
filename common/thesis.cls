% SM Proposal v0.1
% By Avik Pal (avikpal@mit.edu)
% Version 0.1 released 02.22.2022
% Based on https://raw.githubusercontent.com/saurabhg17/SPhdThesis/master/SPhdThesis.cls

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Proposal}[2022/22/02 v0.1 Avik Pal]

\RequirePackage{xkeyval}

% -----------------------------------------------------------------------------
% Define option for formatting for [screen] and print.
% When formatting for screen colors are used for links, tables, and algorithms.
% When formatting for print black color is used for links, tables, and algorithms.
\define@choicekey*[Sg]{PhDThesis}{media}{screen,print}[screen]{\def \SgIntMedia{#1}}
\setkeys[Sg]{PhDThesis}{media}

% -----------------------------------------------------------------------------
% Define option for formatting title page in lower or [upper] case.
\define@choicekey*[Sg]{PhDThesis}{titlecase}{upper,lower}[upper]{\def \SgIntTitleCase{#1}}
\setkeys[Sg]{PhDThesis}{titlecase}

% -----------------------------------------------------------------------------
% Define line spacing. The valid values are [onehalf] and double.
% Note that before using \SgLineSpacing setspace package must be loaded.
\define@choicekey*[Sg]{PhDThesis}{linespacing}[\val\nr]{onehalf,double}[onehalf]{%
  \ifcase\nr\relax
    \def \SgIntLineSpacing{\onehalfspacing}
  \or
    \def \SgIntLineSpacing{\doublespacing}
  \fi
}
\setkeys[Sg]{PhDThesis}{linespacing}

% -----------------------------------------------------------------------------
% Define font size. The valid values are [11pt] and 12pt.
\define@choicekey*[Sg]{PhDThesis}{fontsize}{11pt,12pt,13pt,14pt,15pt}[14pt]{\def \SgIntFontSize{#1}}
\setkeys[Sg]{PhDThesis}{fontsize}

% -----------------------------------------------------------------------------
% Define font family to be used. Default is cm.
\define@choicekey*[Sg]{PhDThesis}{font}{helvet}[helvet]{\def \SgIntFont{#1}}
\setkeys[Sg]{PhDThesis}{font}

% -----------------------------------------------------------------------------
% Define open. The valid values are right and [any].
% Right makes chapters and entries in frontmatter begin  only on right hand pages 
% or on the next page available. Any puts them on the next available page.
\define@choicekey*[Sg]{PhDThesis}{open}[\val\nr]{right,any}[any]{%
  \ifcase\nr\relax
    \def \SgIntOpen {openright}
  \or
    \def \SgIntOpen {openany}
  \fi
}
\setkeys[Sg]{PhDThesis}{open}

% -----------------------------------------------------------------------------
% Define pageside. The valid values are oneside and [twoside].
% Specifies whether double or single sided output should be generated.
\define@choicekey*[Sg]{PhDThesis}{pageside}{oneside,twoside}[twoside]{\def \SgIntPageSide{#1}}
\setkeys[Sg]{PhDThesis}{pageside}


% Process all options defined above with default values.
\ProcessOptionsX[Sg]<PhDThesis>

% Finally load the report class.
\LoadClass[a4paper, \SgIntFontSize, \SgIntOpen, \SgIntPageSide]{report}



% -----------------------------------------------------------------------
% F O N T S.

\RequirePackage[T1]{fontenc}  % Use T1 encoded cm-super fonts.
\RequirePackage{microtype}    % Improve typesetting.
\RequirePackage{fix-cm}       % Support for arbitrary font size for cm.

% Specify the format for the section titles in toc.
\newcommand{\SgIntTocSectionFormat}{\bfseries}

% Specify the format for chapter name and number in chapter headings.
\newcommand{\SgIntChapNameFormat}{\fontsize{20}{50}\fontshape{sc}\selectfont}
\newcommand{\SgIntChapNumberFormat}{\fontsize{76}{80}\selectfont}

% Specify the format for chapter, section, and subsection titles.
\newcommand{\SgIntChapTitleFormat}{\LARGE\bfseries}
\newcommand{\SgIntSecTitleFormat}{\Large\bfseries}
\newcommand{\SgIntSubsecTitleFormat}{\large\bfseries}

% Specify the format for displaying chapter in fancy header.
\newcommand{\SgIntHeaderFormat}{\large\bfseries}
\newcommand{\SgIntPageNumFormat}{\bfseries}


% -----------------------------------------------------------------------------
% C A P T I O N S
% 
% Nicer captions for figures and tables. 
% caption package must be included before subfig and hyperref.

% Change the font of the caption to sans serif and make label bold.
\RequirePackage[labelfont=bf]{caption}



% -----------------------------------------------------------------------
% P A C K A G E S
% 
% Include important packages.

\RequirePackage{graphicx}                             % Figures.
\RequirePackage{amsmath, amssymb, amsthm}             % Math symbols and fonts.
\RequirePackage{float}                                % Must be done before hyperref.
\RequirePackage[usenames,dvipsnames,hyperref,table]{xcolor} % For defining colors.
\RequirePackage{ifthen}                               % For comparison.
\RequirePackage{lipsum}
\RequirePackage{natbib}



% -----------------------------------------------------------------------
% P A G E   L A Y O U T.

% Use geometry package to set up margins.
% A4 paper is 8.27 x 11.69 inch.
\RequirePackage[a4paper, left=1in, right=1in, top=1in, bottom=1in, includehead, \SgIntPageSide]{geometry}

\ifthenelse{\equal{\SgIntOpen}{openright}}
{
  % For adding extra blank page, if necessary, after chapter.
  \RequirePackage{emptypage}

  % Define a command to leave a blank page. This is used in frontmatter to add 
  % an empty page between ack, abstract, toc, lof and lot.
  \newcommand{\SgIntClearDoublePage}{\clearpage{\pagestyle{empty}\cleardoublepage}}
}
{
  \newcommand{\SgIntClearDoublePage}{\clearpage}
}

% For disabling paragraph indenting and using a blank line between paragraphs.
\RequirePackage{parskip}

% Set line spacing.
\RequirePackage{setspace}
\SgIntLineSpacing

% Fix footnote spacing
\setlength{\footnotesep}{0.5cm}   % Distance between two footnotes.
\setlength{\skip\footins}{0.5cm}  % Distance between last line of text and first footnote.

% By default Latex centers images vertically on a float page.
% Modify Latex internal variables so that figures are placed from top.
\makeatletter
\setlength{\@fptop}{0.25cm}
\setlength{\@fpsep}{1.00cm}
\makeatother

\setlength{\textfloatsep}{1.5cm} % Set the distance between a float and text.
\setlength{\floatsep}{1.0cm}     % Set the distance between two floats.

% Prevent latex from adding extra space between paragraphs so that
% the last line is at the bottom margin on each page.
\raggedbottom

% Define horizontal spacing between subfigures.
\newcommand{\SgIntHSpaceBetweenSubfloats}{\hspace{0.5cm}}

\setcounter{secnumdepth}{4}

% -----------------------------------------------------------------------
% T A B L E
% 
% Set table layout and design.

\RequirePackage{booktabs, colortbl} % Tables.
\RequirePackage{tabularx}           % Auto column sizing.

\renewcommand{\arraystretch}{1.5}     % Set space between rows in a table.
\renewcommand{\tabcolsep}{0.20cm}     % Set space between columns in a table.
\heavyrulewidth = 0.15em              % Set width of heavy rules.
\lightrulewidth = 0.07em              % Set width of light rules.
\abovetopsep    = 0.1cm               % Set separation between caption and top rule.
\aboverulesep   = 0.4ex               % Set separation to use above a rule.
\belowrulesep   = 0.4ex               % Set separation to use below a rule.

% Set color for table rules.
\ifthenelse{\equal{\SgIntMedia}{screen}}{\arrayrulecolor[rgb]{0.0, 0.6, 0.9}}{}
\ifthenelse{\equal{\SgIntMedia}{print}}{\arrayrulecolor[rgb]{0.0, 0.0, 0.0}}{}



% -----------------------------------------------------------------------
% A L G O R I T H M

\RequirePackage[algo2e, ruled, linesnumbered, algochapter]{algorithm2e}
\DontPrintSemicolon                      % Dont print semicolons at end of lines.
\algoheightrule       = \heavyrulewidth  % Set the width of the top and bottom rules.
\algotitleheightrule  = \lightrulewidth  % Set the width of the middle rule.
\SetAlgoInsideSkip{medskip}              % Set distance between middle rule and algorithm.
\interspacetitleruled = 0.2cm            % Set distance between caption and rules.
\setlength{\algomargin}{2.25em}          % Set the margin of the algorithm text.
\SetNlSkip{1.25em}                       % Set the spacing between line numbers and text.

\newenvironment{SgAlgorithm}[1][t]
{%
  \begin{algorithm2e}[#1]
    \linespread{1.3} % Set the line spacing to one and half.
    \selectfont      % The linespread is effective only after selectfont.%
    }
    {%
  \end{algorithm2e}
}



% -----------------------------------------------------------------------
% B I B L I O G R A P H Y

% Rename bibliography to references.
\renewcommand{\bibname}{References}

% Define a new command to include the bibliography file and 
% set the formatting options.
\newcommand{\SgIncludeBib}[1]
{%
  \clearpage                                 % Fix the page number in TOC.
  \phantomsection                            % Fix the link in PDF.
  \addcontentsline{toc}{chapter}{References} % Add the bibliography to TOC.
  \bibliographystyle{plainnat}               % Set the bibliography style.
  \bibliography{#1}                          % Include the bibliography file.
}



% -----------------------------------------------------------------------
% F A N C Y   H E A D E R

\RequirePackage{fancyhdr}

% Set page style to fancy.
\pagestyle{fancy}

% By default fancyhdr converts the chapter headings to uppercase,
% so restore the chapter casing.
\renewcommand{\chaptermark}[1]{\markboth{\chaptername\ \thechapter.\ #1}{}}

% Define header consisting of Chapter #. Chapter_Name on left side and 
% page number on right side.
% \nouppercase is used on left header to force bibliography in lowercase.
\fancyhead{}
\fancyhead[L]{\nouppercase{\SgIntHeaderFormat\leftmark}}
\fancyhead[R]{\SgIntPageNumFormat\thepage}
\renewcommand{\headrulewidth}{1pt}
\addtolength{\headheight}{10pt}

% Disable footer.
\fancyfoot{}
\renewcommand{\footrulewidth}{0pt}

% Define header and footer for plain pages.
\fancypagestyle{plain}
{%
  % Disable header.
  \fancyhead{}
  \renewcommand{\headrulewidth}{0pt}

  % Footer contains the page number on right side.
  \fancyfoot{}
  \fancyfoot[R]{\SgIntPageNumFormat\thepage}
  \renewcommand{\footrulewidth}{0pt}
}



% -----------------------------------------------------------------------
% T O C    L O F    L O T    L O A
% 
% Change the appearance of toc, lof, lot, etc.

% Using tocloft, the toc can be formatted easily.
\RequirePackage[titles]{tocloft}

% Remove dots.
\renewcommand{\cftdotsep}{\cftnodots}

% Remove dots from list of algorithms.
% This is necessary because we use algorithm2e which mananges its own list of algorithms.
\makeatletter
\renewcommand{\@dotsep}{5000}
\makeatother

% Format chapter entries differently in toc.
\renewcommand{\cftchapfont}{\SgIntTocSectionFormat}

% Fix the indentation of figure and table entries in the lof, lot, and loa.
\setlength{\cftfigindent}{0in}
\setlength{\cfttabindent}{0in}

\newcommand{\SgAddToc}{\tableofcontents\SgIntClearDoublePage}
\newcommand{\SgAddLof}{%
  \newpage
  \phantomsection % Requires hyperref; this is to fix the link.
  \addcontentsline{toc}{section}{\numberline{}\hspace{-.35in}{\SgIntTocSectionFormat{}List of Figures}}
  \listoffigures
  \SgIntClearDoublePage
}
\newcommand{\SgAddLot}{%
  \newpage
  \phantomsection % Requires hyperref; this is to fix the link.
  \addcontentsline{toc}{section}{\numberline{}\hspace{-.35in}{\SgIntTocSectionFormat{}List of Tables}}
  \listoftables
  \SgIntClearDoublePage
}
\newcommand{\SgAddLoa}{%
  \newpage
  \phantomsection % Requires hyperref; this is to fix the link.
  \addcontentsline{toc}{section}{\numberline{}\hspace{-.35in}{\SgIntTocSectionFormat{}List of Algorithms}}
  \listofalgorithms % Note an extra e, it is required because we use algorithm2e.
  \SgIntClearDoublePage
}



% -------------------------------------------------------------------------------------
% C H A P T E R   H E A D I N G
% 
% Change the appearance of chapter headers and section titles.

% Change the fonts for chapter heading.
\RequirePackage[Sonny]{fncychap}
\ChNameVar{\SgIntChapNameFormat}
\ChNumVar{\SgIntChapNumberFormat}
\ChTitleVar{\SgIntChapTitleFormat}

% Set the format of the section and subsection titles.
\RequirePackage{sectsty}
\sectionfont{\SgIntSecTitleFormat}
\subsectionfont{\SgIntSubsecTitleFormat}

% Adjust the spacing between titles and surrounding matter.
\RequirePackage{etoolbox}
\makeatletter
% Reduce the space between chapter title and text.
\patchcmd{\DOTI}{\vskip 40\p@}{\vskip 20\p@}{}{}
\patchcmd{\DOTIS}{\vskip 40\p@}{\vskip 20\p@}{}{}% for unnumbered chapters

% http://tex.stackexchange.com/questions/13357/fncychap-package-reduce-vertical-gap-space-between-header-and-chapter-heading
% Reduce the spacing between top margin and chapter title.
\renewcommand*{\@makechapterhead}[1]
{%
  \vspace*{10\p@}
  {%
    \parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter % Fix for frontmatter, mainmatter, and backmatter 040920
        \DOCH
      \fi
    \fi
    \interlinepenalty\@M
    \if@mainmatter     % Fix for frontmatter, mainmatter, and backmatter 060424
      \DOTI{#1}%
    \else%      
      \DOTIS{#1}%
    \fi
  }
}

% For the case \chapter*:
\renewcommand*{\@makeschapterhead}[1]
{%
  \vspace*{10\p@}%
  {%
    \parindent \z@ \raggedright \normalfont
    \interlinepenalty\@M \DOTIS{#1} \vskip 10\p@
  }
}
\makeatother



% -----------------------------------------------------------------------
% H Y P E R E F
% 
% Set hyperlink settings.

\RequirePackage[hyphens]{url} % Embedding URL's in document.

% Enable hyperlinks only in the PDF.
\RequirePackage[bookmarks=true, pdfstartview=Fit, linktoc=page, pdfpagemode=UseNone, backref=page]{hyperref}

% Define the colors of the hyperlink.
\definecolor{webgreen}{rgb}{0,.5,0} % previously used magenta.
\definecolor{webblue}{rgb}{0,0,1}   % previously used orange.

\hypersetup
{%
  pdftitle = On Efficient Training and Inference of Neural Differential Equations,
  pdfauthor = Avik Pal,
  colorlinks = true,
  linkcolor  = webgreen,
  citecolor  = magenta,
  urlcolor   = webblue
}

\usepackage{etoolbox}
\makeatletter
\patchcmd{\BR@backref}{\newblock}{\newblock(page~}{}{}
\patchcmd{\BR@backref}{\par}{)\par}{}{}
\makeatother


% Define the proerties describing PDF.
\newcommand{\SgIntSetupPdfProps}%
{%
  \hypersetup
  {%
    pdfauthor   = \SgIntAuthor,
    pdftitle    = \SgIntTitle,
    pdfsubject  = \SgIntSubject,
    pdfkeywords = \SgIntKeywords
  }
}



% -----------------------------------------------------------------------
% D O C U M E N T   P R O P E R T I E S
% 
% Important variables used for creating PDF properties and the title page.

% Define empty variables for creating title and setting PDF properties.
\newcommand{\SgIntTitle}{}
\newcommand{\SgIntAuthor}{}
\newcommand{\SgIntAuthorDegrees}{}
\newcommand{\SgIntYear}{}
\newcommand{\SgIntSubject}{}
\newcommand{\SgIntKeywords}{}
\newcommand{\SgIntDegree}{}
\newcommand{\SgIntDepartment}{}
\newcommand{\SgIntUniversity}{}
\newcommand{\SgIntDeclarationDate}{}
\newcommand{\SgIntCompletionDate}{}
\newcommand{\SgIntSubmissionDate}{}
\newcommand{\SgIntLab}{}
\newcommand{\SgIntAddrLineOne}{}
\newcommand{\SgIntAddrLineTwo}{}
\newcommand{\SgIntBriefProblemStatement}{}
\newcommand{\SgIntAdvisor}{}
\newcommand{\SgIntAdvisorRole}{}
\newcommand{\SgIntAdvisorDept}{}
\newcommand{\SgIntTitleSplitOne}{}
\newcommand{\SgIntTitleSplitTwo}{}

% Define commands for setting  the above variables.
\newcommand{\SgSetTitle}[1]{\renewcommand{\SgIntTitle}{#1}\SgIntSetupPdfProps}
\newcommand{\SgSetAuthor}[1]{\renewcommand{\SgIntAuthor}{#1}\SgIntSetupPdfProps}
\newcommand{\SgSetAuthorDegrees}[1]{\renewcommand{\SgIntAuthorDegrees}{#1}}
\newcommand{\SgSetYear}[1]{\renewcommand{\SgIntYear}{#1}}
\newcommand{\SgSetSubject}[1]{\renewcommand{\SgIntSubject}{#1}\SgIntSetupPdfProps}
\newcommand{\SgSetKeywords}[1]{\renewcommand{\SgIntKeywords}{#1}\SgIntSetupPdfProps}
\newcommand{\SgSetDegree}[1]{\renewcommand{\SgIntDegree}{#1}}
\newcommand{\SgSetDepartment}[1]{\renewcommand{\SgIntDepartment}{#1}}
\newcommand{\SgSetUniversity}[1]{\renewcommand{\SgIntUniversity}{#1}}
\newcommand{\SgSetDeclarationDate}[1]{\renewcommand{\SgIntDeclarationDate}{#1}}
\newcommand{\SgSetCompletionDate}[1]{\renewcommand{\SgIntCompletionDate}{#1}}
\newcommand{\SgSetSubmissionDate}[1]{\renewcommand{\SgIntSubmissionDate}{#1}}
\newcommand{\SgSetLab}[1]{\renewcommand{\SgIntLab}{#1}}
\newcommand{\SgSetAddrLineOne}[1]{\renewcommand{\SgIntAddrLineOne}{#1}}
\newcommand{\SgSetAddrLineTwo}[1]{\renewcommand{\SgIntAddrLineTwo}{#1}}
\newcommand{\SgSetBriefProblemStatement}[1]{\renewcommand{\SgIntBriefProblemStatement}{#1}}
\newcommand{\SgSetAdvisor}[1]{\renewcommand{\SgIntAdvisor}{#1}}
\newcommand{\SgSetAdvisorRole}[1]{\renewcommand{\SgIntAdvisorRole}{#1}}
\newcommand{\SgSetAdvisorDept}[1]{\renewcommand{\SgIntAdvisorDept}{#1}}
\newcommand{\SgSetTitleSplitOne}[1]{\renewcommand{\SgIntTitleSplitOne}{#1}}
\newcommand{\SgSetTitleSplitTwo}[1]{\renewcommand{\SgIntTitleSplitTwo}{#1}}



% -----------------------------------------------------------------------
% C L E V E R E F
% 
% Must come as late as possible, especially after hyperref.
\RequirePackage[capitalize]{cleveref}

% Disable the automatic abbreviations of equations and figures.
\crefname{equation}{Equation}{Equations}
\crefname{figure}{Figure}{Figures}
\Crefname{equation}{Equation}{Equations}
\Crefname{figure}{Figure}{Figures}

% Change the way links are produced in PDF documents.
\crefformat{chapter}{#2Chapter~#1#3}
\crefformat{section}{#2Section~#1#3}
\crefformat{figure}{#2Figure~#1#3}
\crefformat{equation}{#2Equation~#1#3}
\crefformat{table}{#2Table~#1#3}
\Crefformat{chapter}{#2Chapter~#1#3}
\Crefformat{section}{#2Section~#1#3}
\Crefformat{figure}{#2Figure~#1#3}
\Crefformat{equation}{#2Equation~#1#3}
\Crefformat{table}{#2Table~#1#3}
\creflabelformat{equation}{#2#1#3}



% -----------------------------------------------------------------------
% T I T L E  P A G E
\ifthenelse{\equal{\SgIntTitleCase}{upper}}
{
  \newcommand{\SgIntMakeUpperCase}[1]{\MakeUppercase{#1}}
}
{
  \newcommand{\SgIntMakeUpperCase}[1]{#1}
}
\newcommand{\SgAddTitle}{%
  \thispagestyle{empty}
  \begin{center}
    % \vspace*{-1.0in}
    \Large{\bf{\SgIntMakeUpperCase{\SgIntTitle}}}\\
    \vspace{0.25in}
    \large{By}\\
    \vspace{0.15in}
    \large{\SgIntMakeUpperCase{\SgIntAuthor}}\\
    \vspace{0.10in}
    \large{B.Tech., Indian Institute of Technology Kanpur (2021)}\\
    \vspace{0.25in}

    \large{Submitted to the \SgIntDepartment~in Partial}\\
    \large{Fulfillment of the Requirements for the Degree of}\\
    \vspace{0.15in}
    \large{\SgIntMakeUpperCase{\SgIntDegree}}\\
    \large{at the}\\
    \large{\SgIntMakeUpperCase{Massachusetts Institute of Technology}}\\
    \vspace{0.15in}
    \large{\SgIntMakeUpperCase{June 2023}}\\
    \vspace{0.25in}
    \small{\copyright 2023 Avik Pal. The work is licensed under \href{https://creativecommons.org/licenses/by-sa/2.0/}{CC BY-SA 2.0}.\\The author hereby grants to MIT a nonexclusive, worldwide, irrevocable, royalty-free license to exercise any and all rights under copyright, including to reproduce, preserve, distribute and publicly display copies of the thesis, or release the thesis under an open-access license.}
    \vspace{0.4in}
  \end{center}

  \large{Authored By: \phantom{xxxxxx} \SgIntAuthor}\\
  \large{\phantom{Authored By: xxxxxx} \SgIntDepartment}\\
  \large{\phantom{Authored By: xxxxxx} \today}\\
  \vspace{0.05in}\\
  \large{Certified By: \phantom{xxxxxxx} \SgIntAdvisor}\\
  \large{\phantom{Certified By: xxxxxxx} Professor of Applied Mathematics}\\
  \large{\phantom{Certified By: xxxxxxx} Thesis Supervisor}\\
  % \vspace{0.02in}\\
  % \large{Certified By: \phantom{xxxxxxx} Chris Rackauckas}\\
  % \large{\phantom{Certified By: xxxxxxx} Research Affiliate, Department of Applied Mathematics}\\
  % \large{\phantom{Certified By: xxxxxxx} Thesis Supervisor}\\
  \vspace{0.05in}\\
  \large{Accepted By: \phantom{xxxxxxx} Leslie A. Kolodziejski}\\
  \large{\phantom{Accepted By: xxxxxxx} Professor of Electrical Engineering and Computer Science}\\
  \large{\phantom{Accepted By: xxxxxxx} Chair, Department Committee on Graduate Students}
  \SgIntClearDoublePage
}



% -----------------------------------------------------------------------
% A C K N O W L E D G M E N T S
% 
% The acknowledgments environment puts a large, bold, centered 
% "Acknowledgments" label at the top of the page. The acknowledgments
% themselves appear in a quote environment, i.e. tabbed in at both 
% sides, and on its own page.
\newenvironment{acknowledgments}%
{%
  \thispagestyle{plain}
  \begin{center}
    \vspace*{1.5cm}
    {%
      \Large \bfseries Acknowledgments
    }
  \end{center}
  \vspace{0.5cm}
  \begin{quote}
    }
    {%
  \end{quote}
  \SgIntClearDoublePage
}



% -----------------------------------------------------------------------
% A B S T R A C T
% 
% The abstract environment puts a large, bold, centered "Abstract" label at
% the top of the page. The abstract itself appears in a quote environment,
% i.e. tabbed in at both sides, and on its own page.
\renewenvironment{abstract}
{%
  \thispagestyle{plain}

  \begin{center}
    {\Large\bfseries \SgIntTitle}\\
    \vspace{0.5em}
    {by}\\
    \vspace{0.5em}
    {\Large\bfseries \SgIntAuthor}\\
    \vspace{1.0em}
    \large{Submitted to the \SgIntDepartment{}}\\
    \large{on \today{} in partial fulfillment of the requirements for the degree of}\\
    \large{\SgIntDegree{} in Electrical Engineering and Computer Science}\\
  \end{center}

  \vspace{1.5em}
  {\Large\bfseries Abstract}\\
  \vspace{0.25em}
  \begin{quote}
  }
  {%
  \end{quote}

  \vspace{1.5em}
  \large{\textit{Thesis Supervisor:} \SgIntAdvisor}\\
  \large{\textit{Title:} Professor of Applied Mathematics}
  % \vspace{0.25em}\\
  % \large{\textit{Thesis Supervisor:} Chris Rackauckas}\\
  % \large{\textit{Title:} Research Affiliate, Department of Applied Mathematics}

  \SgIntClearDoublePage
}

% -----------------------------------------------------------------------
% SECTION HEADINGS: CAPITALIZE
% 
\RequirePackage{titlesec}

\makeatletter

\titleformat{\section}{\normalfont\Large\scshape\bfseries}{\thesection}{.5em}{}[]

\titleformat{\subsection}{\normalfont\large\scshape\bfseries}{\thesubsection}{.5em}{}[]

\titleformat{\subsubsection}{\normalfont\normalsize\scshape\bfseries}{\thesubsubsection}{.5em}{}[]

\makeatother




% -----------------------------------------------------------------------
% F R O N T M A T T E R
% 
% The frontmatter environment set the page numbering to lowercase roman for 
% ack, abstract, toc, lof, lot, loa, etc. It also resets page numbering for the 
% remainder of thesis (arabic, starting at 1).
\newenvironment{frontmatter}
{%
  \setcounter{page}{1}
  \renewcommand{\thepage}{\roman{page}}
}
{%
  \clearpage
  \renewcommand{\thepage}{\arabic{page}}
  \setcounter{page}{1}
  \SgIntClearDoublePage
}