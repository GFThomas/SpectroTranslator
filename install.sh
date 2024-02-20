#! /bin/bash -f



## Chech that the PYTHON packages are installed
echo " "
echo " "
echo "INSTALL PYTHON PACKAGES"
echo " "
package_list="numpy astropy matplotlib scikit-learn==1.1.1 shap pickle tensorflow==2.11.0 tensorflow-addons keras==2.11.0 pandas scipy"

python3 -m pip install ${package_list}

echo ""
echo "---------------------------------------"
echo "         INSTALLATION DONE"
echo "---------------------------------------"