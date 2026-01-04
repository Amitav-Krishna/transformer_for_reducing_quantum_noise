\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{myclass}[2025/11/13 Custom Org override class]

% Load base class
\LoadClass[final]{article}

% Your packages
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{hyperref}

% ICML style
\RequirePackage{icml2025}

% BEGIN DOCUMENT INSIDE CLASS
\AtEndOfClass{
  % User will provide title/affils via \icmltitle etc.
  \begin{document}
}
