;--------------------------------
; Includes

  !include "MUI2.nsh"
  !include "logiclib.nsh"

;--------------------------------
; Custom defines
  !define NAME "REAL Video Enhancer"
  !define APPFILE "REAL-Video-Enhancer.exe"
  !define VERSION "2.3.8"
  !define SLUG "${NAME} v${VERSION}"
  !define COMPANYNAME "TNTwise"
  !define VERSIONMAJOR 2
  !define VERSIONMINOR 3
  !define VERSIONBUILD 8 
  !define DISPLAYVERSION "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}"
  !define INSTALLSIZE 297000


;--------------------------------
; General

  Name "${NAME}"
  OutFile "REAL-Video-Enhancer-${DISPLAYVERSION}-Windows-Setup.exe"
  InstallDir "$PROGRAMFILES\${NAME}"
  InstallDirRegKey HKCU "Software\${NAME}" ""
  RequestExecutionLevel admin


; UI
  
  !define MUI_ICON "icons/logo-v2.ico"
  !define MUI_WELCOMEPAGE_TITLE "${SLUG} Setup"

;--------------------------------
; Pages
  
  ; Installer pages
  !insertmacro MUI_PAGE_WELCOME
  !insertmacro MUI_PAGE_LICENSE "LICENSE"
  !insertmacro MUI_PAGE_DIRECTORY
  !insertmacro MUI_PAGE_INSTFILES
  !insertmacro MUI_PAGE_FINISH

  ; Uninstaller pages
  !insertmacro MUI_UNPAGE_CONFIRM
  !insertmacro MUI_UNPAGE_INSTFILES
  
  ; Set UI language
  !insertmacro MUI_LANGUAGE "English"

;--------------------------------
; Section - Install App

Section "install"
    SectionIn RO
    SetOutPath "$INSTDIR"
    File /r "dist\REAL-Video-Enhancer\*.*" 
    File /r "icons\logo-v2.ico" 
    createDirectory "$COMMONSMPROGRAMS\${COMPANYNAME}"
	  createShortCut "$COMMONSMPROGRAMS\${COMPANYNAME}\${NAME}.lnk" "$INSTDIR\REAL-Video-Enhancer.exe" "" "$INSTDIR\logo-v2.ico"
    writeUninstaller "$INSTDIR\Uninstall.exe"
    RMDir /r "$INSTDIR\backend" 
    # Registry information for add/remove programs
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}" "DisplayName" "${NAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}" "QuietUninstallString" "$\"$INSTDIR\uninstall.exe$\" /S"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}" "InstallLocation" "$\"$INSTDIR$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}" "DisplayIcon" "$\"$INSTDIR\logo-v2.ico$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}" "DisplayVersion" "$\"${DISPLAYVERSION}$\""
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}" "VersionMajor" ${VERSIONMAJOR}
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}" "VersionMinor" ${VERSIONMINOR}
    # There is no option for modifying or repairing the install
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}" "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}" "NoRepair" 1
    # Set the INSTALLSIZE constant (!defined at the top of this script) so Add/Remove Programs can accurately report the size
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}" "EstimatedSize" ${INSTALLSIZE}
SectionEnd

;--------------------------------
; Section - Shortcut

  Section "Desktop Shortcut" DeskShort
    CreateShortCut "$COMMONDESKTOP\${NAME}.lnk" "$INSTDIR\${APPFILE}"
  SectionEnd

;--------------------------------
; Remove empty parent directories

  Function un.RMDirUP
    !define RMDirUP '!insertmacro RMDirUPCall'

    !macro RMDirUPCall _PATH
          push '${_PATH}'
          Call un.RMDirUP
    !macroend

    ; $0 - current folder
    ClearErrors

    Exch $0
    ;DetailPrint "ASDF - $0\.."
    RMDir "$0\.."

    IfErrors Skip
    ${RMDirUP} "$0\.."
    Skip:

    Pop $0

  FunctionEnd

;--------------------------------
; Section - Uninstaller

Section "Uninstall"
  
  ;Delete Shortcut
  Delete "$COMMONDESKTOP\${NAME}.lnk"
  Delete "$COMMONSMPROGRAMS\${COMPANYNAME}\${NAME}.lnk"
  RMDir "$COMMONSMPROGRAMS\${COMPANYNAME}"

  ;Delete Uninstall
  Delete "$INSTDIR\Uninstall.exe"

  ;Delete Folder
  RMDir /r "$INSTDIR"
  ${RMDirUP} "$INSTDIR"
  RMDir /r "$LOCALAPPDATA\REAL-Video-Enhancer"

  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${COMPANYNAME} ${NAME}"

SectionEnd
