#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<stdint.h>
#include<stdbool.h>

typedef struct {
     unsigned char gray;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

#define CREATOR "Fingerprint_23_26_33_36_40"
#define GRAY_COMPONENT_COLOR 255
#define DBL_EPSILON 2.2204460492503131e-16
#define PI 3.142857

//read image
static PPMImage *readPPM(const char *filename)
{
         char buff[16];
         PPMImage *img;
         FILE *fp;
         int c, gray_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '5') {
         fprintf(stderr, "Invalid image format (must be 'P5')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &gray_comp_color) != 1) {
         fprintf(stderr, "Invalid gray component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (gray_comp_color!= GRAY_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel*)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    if (fread(img->data, img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}

//write image
void writePPM(const char *filename, PPMImage *img)
{
    FILE *fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P5\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",GRAY_COMPONENT_COLOR);

    // pixel data
    fwrite(img->data, img->x, img->y, fp);
    fclose(fp);
}

//Stores image data to uint8_t array pointer
void imageCopy(uint8_t *imagedata, PPMImage *image,int width, int height){
    int i;
    for (i=0;i<width*height;i++){
        imagedata[i] = (uint8_t)image->data[i].gray;
    }
}

//Enhances the image contrast using histogram approach
void histogramAnalysis(uint8_t *image, int width, int height){
    int hist[256] = { 0 };
    int row, col, intmin, intmax, intsum, percent=1;

    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            hist[(int)image[row*width+col]]+=1;
        }
    }

    intsum=0;
    for(intmin=0;intmin<256;intmin++){
        intsum+=hist[intmin];
        if ( ( (intsum*100) / (width*height) ) > percent ){
            break;
        }
    }

    intsum=0;
    for(intmax=255;intmax>=0;intmax--){
        intsum+=hist[intmax];
        if ( ( (intsum*100) / (width*height) ) > percent ){
            break;
        }
    }

    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if(image[row*width+col]<intmin){
                image[row*width+col]=0;
            }
            else if (image[row*width+col]>intmax){
                image[row*width+col]=255;
            }
            else {
                image[row*width+col]=(uint8_t)(255.0*((int)image[row*width+col]-intmin)/(intmax-intmin)+0.5);
            }
        }
    }
}

void imageDouble(double *imagedata, uint8_t *im, int width, int height){
    int i;
    for (i=0;i<width*height;i++){
        imagedata[i] = (double)im[i];
    }
}

void segmentation(double *image, bool *mask, int width, int height){
    int row, col, new_row, new_col, blocksize=3;
    double dblstddv, dblmean, dblsum=0.0;
    double thresh=(-0.1);

    //padded image dimensions
    new_row = (int)(blocksize*ceil((float)height/(float)blocksize));
    new_col = (int)(blocksize*ceil((float)width/(float)blocksize));

    double *paddedim = (double*)malloc(new_row*new_col*sizeof(double));
    double *stddevim = (double*)malloc(new_row*new_col*sizeof(double));

    //zero padding
    for(row=0;row<new_row;row++){
        for(col=0;col<new_col;col++){
            stddevim[row*new_col+col] = 0.0;
            if (row<height && col<width){
                paddedim[row*new_col+col] = image[row*width+col];
            }
            else{
                paddedim[row*new_col+col] = 0.0;
            }
        }
    }

    //Local analysis by standard deviation
    for(row=0;row<new_row;row=row+blocksize){
        for(col=0;col<new_col;col=col+blocksize){
            int blockrow, blockcol;

            //mean
            dblsum=0.0;
            for(blockrow=row;blockrow<row+blocksize;blockrow++){
                for(blockcol=col; blockcol<col+blocksize;blockcol++){
                    dblsum+=paddedim[blockrow*new_col+blockcol];
                }
            }
            dblmean=dblsum/(double)(blocksize*blocksize);

            //standard deviation
            dblsum=0.0;
            for(blockrow=row;blockrow<row+blocksize;blockrow++){
                for(blockcol=col; blockcol<col+blocksize;blockcol++){
                    dblsum+=pow((paddedim[blockrow*new_col+blockcol]-dblmean),2);
                }
            }
            dblstddv=sqrt(dblsum/(double)(blocksize*blocksize));

            //assigning std deviation
            for(blockrow=row;blockrow<row+blocksize;blockrow++){
                for(blockcol=col; blockcol<col+blocksize;blockcol++){
                    stddevim[blockrow*new_col+blockcol]=dblstddv;
                }
            }
        }
    }
    
    //segmenting image by standard deviation threshold
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if (stddevim[row*width+col]>thresh){
                mask[row*width+col]=true;
            }
            else{
                mask[row*width+col]=false;
            }
        }
    }
}

void normalization(double *image, double *normIm, bool *mask, int width, int height){
    int row, col, total=0;
    double dblsum, dblmean, dblstddev;

    //mean
    dblsum=0.0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if (mask[row*width+col]==true){
                dblsum+=image[row*width+col];
                total++;
            }
        }
    }
    dblmean=dblsum/(double)total;

    //standard deviation
    dblsum=0.0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if (mask[row*width+col]==true){
                dblsum+=pow((image[row*width+col]-dblmean),2);
            }
        }
    }
    dblstddev = sqrt(dblsum/(double)total);

    //normalization
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            normIm[row*width+col] = (image[row*width+col] - dblmean)/dblstddev;
        }
    }
}

void createGaussianKernelSigma(double *GKernel, int sigma, int kernelSize){
    int x,y,i,j;
    double r, s = 2.0 * sigma * sigma;
    double gsum = 0.0;
    
    //creating kernel
    for (x = -(kernelSize/2); x <= (kernelSize/2); x++) {
        for (y = -(kernelSize/2); y <= (kernelSize/2); y++) {
            r = sqrt(x * x + y * y);
            GKernel[(x + (kernelSize/2))*kernelSize+(y + (kernelSize/2))] = (exp(-(r * r) / s)) / (PI * s);
            gsum += GKernel[(x + (kernelSize/2))*kernelSize+(y + (kernelSize/2))];
        }
    }
    
    // normalising the Kernel
    for (i = 0; i < kernelSize; ++i){
        for (j = 0; j < kernelSize; ++j){
            GKernel[i*kernelSize+j] /= gsum;
        }
    }
}

void creategradientXgradientY(double *gradientX, double *gradientY, double *GKernel, int kernelSize){
    int i,j;

    for (i=0;i<kernelSize;i++){
        for(j=0;j<kernelSize;j++){
            if (i==0){
                gradientY[i*kernelSize+j]=GKernel[(i+1)*kernelSize+j]-GKernel[i*kernelSize+j];
            }
            else if (i==(kernelSize-1)){
                gradientY[i*kernelSize+j]=GKernel[i*kernelSize+j]-GKernel[(i-1)*kernelSize+j];
            }
            else{
                gradientY[i*kernelSize+j]=(GKernel[(i+1)*kernelSize+j]-GKernel[(i-1)*kernelSize+j])/2.0;
            }

            if (j==0){
                gradientX[i*kernelSize+j]=GKernel[i*kernelSize+(j+1)]-GKernel[i*kernelSize+j];
            }
            else if (j==(kernelSize-1)){
                gradientX[i*kernelSize+j]=GKernel[i*kernelSize+j]-GKernel[i*kernelSize+(j-1)];
            }
            else{
                gradientX[i*kernelSize+j]=(GKernel[i*kernelSize+(j+1)]-GKernel[i*kernelSize+(j-1)])/2.0;
            }
        }
    }
}

void paddingZeroesOnEdges(double *paddedNormIm, double *normIm, int kernelSize, int gradientPaddedWidth, int gradientPaddedHeight, int width, int height){
    int row,col;
    for (row=0;row<gradientPaddedHeight;row++){
        for(col=0;col<gradientPaddedWidth;col++){
            if (row<(kernelSize/2) || col<(kernelSize/2) || col>=(gradientPaddedWidth-(kernelSize/2)) || row>=(gradientPaddedHeight-(kernelSize/2))){
                paddedNormIm[row*gradientPaddedWidth+col]=0.0;
            }
            else{
                paddedNormIm[row*gradientPaddedWidth+col]=normIm[(row-(kernelSize/2))*width+(col-(kernelSize/2))];
            }
        }
    }
}

void convolve2d(double *paddedNormIm, double *gradientX, double *gradientY, double *Gx, double *Gy, int kernelSize, int gradientPaddedWidth, int width, int height){
    int row, col, i, j;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            double sumx=0.0;
            double sumy=0.0;
            for (i=-(kernelSize/2);i<=(kernelSize/2);i++){
                for(j=-(kernelSize/2);j<=(kernelSize/2);j++){
                    sumx+=paddedNormIm[(row+i+(kernelSize/2))*gradientPaddedWidth+(col+j+(kernelSize/2))]*gradientX[(i+(kernelSize/2))*kernelSize+(j+(kernelSize/2))];
                    sumy+=paddedNormIm[(row+i+(kernelSize/2))*gradientPaddedWidth+(col+j+(kernelSize/2))]*gradientY[(i+(kernelSize/2))*kernelSize+(j+(kernelSize/2))];
                }
            }
            Gx[row*width+col]=sumx;
            Gy[row*width+col]=sumy;
        }
    }
}

void createGxxGyyGxy(double *Gxx, double *Gyy, double *Gxy, double *Gx, double *Gy, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            Gxx[row*width+col]=pow(Gx[row*width+col],2);
            Gyy[row*width+col]=pow(Gy[row*width+col],2);
            Gxy[row*width+col]=Gx[row*width+col]*Gy[row*width+col];
        }
    }
}

void paddingReflectGradient(double *paddedGxx, double *Gxx, double *paddedGyy, double *Gyy, double *paddedGxy, double *Gxy, int kernelSize, int gradientPaddedWidth, int gradientPaddedHeight, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            paddedGxx[(row+(kernelSize/2))*gradientPaddedWidth+(col+(kernelSize/2))]=Gxx[row*width+col];
            paddedGyy[(row+(kernelSize/2))*gradientPaddedWidth+(col+(kernelSize/2))]=Gyy[row*width+col];
            paddedGxy[(row+(kernelSize/2))*gradientPaddedWidth+(col+(kernelSize/2))]=Gxy[row*width+col];
        }
    }
    for(row=(kernelSize/2);row<(gradientPaddedHeight-(kernelSize/2));row++){
        for(col=0;col<(kernelSize/2);col++){
            paddedGxx[row*gradientPaddedWidth+col]=paddedGxx[row*gradientPaddedWidth+(kernelSize-2-col)];
            paddedGyy[row*gradientPaddedWidth+col]=paddedGyy[row*gradientPaddedWidth+(kernelSize-2-col)];
            paddedGxy[row*gradientPaddedWidth+col]=paddedGxy[row*gradientPaddedWidth+(kernelSize-2-col)];
        }
        for(col=(gradientPaddedWidth-1);col>=(gradientPaddedWidth-(kernelSize/2));col--){
            paddedGxx[row*gradientPaddedWidth+col]=paddedGxx[row*gradientPaddedWidth+((gradientPaddedWidth-kernelSize)+(gradientPaddedWidth-col))];
            paddedGyy[row*gradientPaddedWidth+col]=paddedGyy[row*gradientPaddedWidth+((gradientPaddedWidth-kernelSize)+(gradientPaddedWidth-col))];
            paddedGxy[row*gradientPaddedWidth+col]=paddedGxy[row*gradientPaddedWidth+((gradientPaddedWidth-kernelSize)+(gradientPaddedWidth-col))];
        }
    }
    for(col=0;col<gradientPaddedWidth;col++){
        for(row=0;row<(kernelSize/2);row++){
            paddedGxx[row*gradientPaddedWidth+col]=paddedGxx[(kernelSize-2-row)*gradientPaddedWidth+col];
            paddedGyy[row*gradientPaddedWidth+col]=paddedGyy[(kernelSize-2-row)*gradientPaddedWidth+col];
            paddedGxy[row*gradientPaddedWidth+col]=paddedGxy[(kernelSize-2-row)*gradientPaddedWidth+col];
        }
        for(row=(gradientPaddedHeight-1);row>=(gradientPaddedHeight-(kernelSize/2));row--){
            paddedGxx[row*gradientPaddedWidth+col]=paddedGxx[((gradientPaddedHeight-kernelSize)+(gradientPaddedHeight-row))*gradientPaddedWidth+col];
            paddedGyy[row*gradientPaddedWidth+col]=paddedGyy[((gradientPaddedHeight-kernelSize)+(gradientPaddedHeight-row))*gradientPaddedWidth+col];
            paddedGxy[row*gradientPaddedWidth+col]=paddedGxy[((gradientPaddedHeight-kernelSize)+(gradientPaddedHeight-row))*gradientPaddedWidth+col];
        }
    }
}

void ndConvolutionGradient(double *paddedGxx, double *paddedGyy, double *paddedGxy, double *Gxx, double *Gyy, double *Gxy, double *GKernelBlock, int gradientPaddedWidth, int kernelSize, int width, int height){
    int row, col, i, j;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            double sumxx=0.0;
            double sumyy=0.0;
            double sumxy=0.0;
            for (i=-(kernelSize/2);i<=(kernelSize/2);i++){
                for(j=-(kernelSize/2);j<=(kernelSize/2);j++){
                    sumxx+=paddedGxx[(row+i+(kernelSize/2))*gradientPaddedWidth+(col+j+(kernelSize/2))]*GKernelBlock[(i+(kernelSize/2))*kernelSize+(j+(kernelSize/2))];
                    sumyy+=paddedGyy[(row+i+(kernelSize/2))*gradientPaddedWidth+(col+j+(kernelSize/2))]*GKernelBlock[(i+(kernelSize/2))*kernelSize+(j+(kernelSize/2))];
                    sumxy+=paddedGxy[(row+i+(kernelSize/2))*gradientPaddedWidth+(col+j+(kernelSize/2))]*GKernelBlock[(i+(kernelSize/2))*kernelSize+(j+(kernelSize/2))];
                }
            }
            Gxx[row*width+col]=sumxx;
            Gyy[row*width+col]=sumyy;
            Gxy[row*width+col]=2*sumxy;
        }
    }
}

void createSinCos(double *sin2theta, double *cos2theta, double *denom, double *Gxx, double *Gyy, double *Gxy, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            denom[row*width+col] = sqrt(pow(Gxy[row*width+col],2)+pow((Gxx[row*width+col]-Gyy[row*width+col]),2)) + DBL_EPSILON;
            sin2theta[row*width+col] = Gxy[row*width+col]/denom[row*width+col];
            cos2theta[row*width+col] = (Gxx[row*width+col]-Gyy[row*width+col])/denom[row*width+col];
        }
    }
}

void paddingReflectSinCos(double *paddedsin2theta, double *paddedcos2theta, double *sin2theta, double *cos2theta, int kernelSize, int gradientPaddedWidth, int gradientPaddedHeight, int width, int height){
    int row,col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            paddedsin2theta[(row+(kernelSize/2))*gradientPaddedWidth+(col+(kernelSize/2))]=sin2theta[row*width+col];
            paddedcos2theta[(row+(kernelSize/2))*gradientPaddedWidth+(col+(kernelSize/2))]=cos2theta[row*width+col];
        }
    }
    for(row=(kernelSize/2);row<(gradientPaddedHeight-(kernelSize/2));row++){
        for(col=0;col<(kernelSize/2);col++){
            paddedsin2theta[row*gradientPaddedWidth+col]=paddedsin2theta[row*gradientPaddedWidth+(kernelSize-2-col)];
            paddedcos2theta[row*gradientPaddedWidth+col]=paddedcos2theta[row*gradientPaddedWidth+(kernelSize-2-col)];
        }
        for(col=(gradientPaddedWidth-1);col>=(gradientPaddedWidth-(kernelSize/2));col--){
            paddedsin2theta[row*gradientPaddedWidth+col]=paddedsin2theta[row*gradientPaddedWidth+((gradientPaddedWidth-kernelSize)+(gradientPaddedWidth-col))];
            paddedcos2theta[row*gradientPaddedWidth+col]=paddedcos2theta[row*gradientPaddedWidth+((gradientPaddedWidth-kernelSize)+(gradientPaddedWidth-col))];
        }
    }
    for(col=0;col<gradientPaddedWidth;col++){
        for(row=0;row<(kernelSize/2);row++){
            paddedsin2theta[row*gradientPaddedWidth+col]=paddedsin2theta[(kernelSize-2-row)*gradientPaddedWidth+col];
            paddedcos2theta[row*gradientPaddedWidth+col]=paddedcos2theta[(kernelSize-2-row)*gradientPaddedWidth+col];
        }
        for(row=(gradientPaddedHeight-1);row>=(gradientPaddedHeight-(kernelSize/2));row--){
            paddedsin2theta[row*gradientPaddedWidth+col]=paddedsin2theta[((gradientPaddedHeight-kernelSize)+(gradientPaddedHeight-row))*gradientPaddedWidth+col];
            paddedcos2theta[row*gradientPaddedWidth+col]=paddedcos2theta[((gradientPaddedHeight-kernelSize)+(gradientPaddedHeight-row))*gradientPaddedWidth+col];
        }
    }
}

void ndConvolutionSinCos(double *paddedsin2theta, double *paddedcos2theta, double *GKernelOrient, double *sin2theta, double *cos2theta, int gradientPaddedWidth, int kernelSize, int width, int height){
    int row, col, i, j;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            double sumsin=0.0;
            double sumcos=0.0;
            for (i=-(kernelSize/2);i<=(kernelSize/2);i++){
                for(j=-(kernelSize/2);j<=(kernelSize/2);j++){
                    sumsin+=paddedsin2theta[(row+i+(kernelSize/2))*gradientPaddedWidth+(col+j+(kernelSize/2))]*GKernelOrient[(i+(kernelSize/2))*kernelSize+(j+(kernelSize/2))];
                    sumcos+=paddedcos2theta[(row+i+(kernelSize/2))*gradientPaddedWidth+(col+j+(kernelSize/2))]*GKernelOrient[(i+(kernelSize/2))*kernelSize+(j+(kernelSize/2))];
                }
            }
            sin2theta[row*width+col]=sumsin;
            cos2theta[row*width+col]=sumcos;
        }
    }
}

void createOrientIm(double *orientIm, double *sin2theta, double *cos2theta, int width, int height){
    int row,col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            orientIm[row*width+col] = (PI/2.0) + (atan2(sin2theta[row*width+col],cos2theta[row*width+col])/2.0);
        }
    }
}

double ridgeFrequency(double *normIm, double *orientIm, bool *mask, double *freqIm, int freqBlockSize, int freqWindowSize, int minWavelength, int maxWavelength, int width, int height){
    int row,col,i,j;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            freqIm[row*width+col]=0.0;
        }
    }
    for(row=0;row<height-freqBlockSize;row+=freqBlockSize){
        for(col=0;col<width-freqBlockSize;col+=freqBlockSize){
            double *blockimagef=(double*)malloc(freqBlockSize*freqBlockSize*sizeof(double));
            double *orientblock=(double*)malloc(freqBlockSize*freqBlockSize*sizeof(double));
            double sinorient, cosorient, orientangle;

            for (i=0;i<freqBlockSize;i++){
                for(j=0;j<freqBlockSize;j++){
                    blockimagef[i*freqBlockSize+j]=normIm[(row+i)*width+(col+j)];
                    orientblock[i*freqBlockSize+j]=orientIm[(row+i)*width+(col+j)];
                }
            }

            // sinorient=0.0;
            // cosorient=0.0;
            // for (i=0;i<freqBlockSize;i++){
            //     for(j=0;j<freqBlockSize;j++){
            //         sinorient+=sin(orientblock[i*freqBlockSize+j]);
            //         cosorient+=cos(orientblock[i*freqBlockSize+j]);
            //     }
            // }
            // sinorient/=(double)(freqBlockSize*freqBlockSize);
            // cosorient/=(double)(freqBlockSize*freqBlockSize);
            // orientangle = atan2(sinorient, cosorient)/2;
            // double angle=orientangle/PI *180 + 90;

            // double *blockimagefindex=(double *)malloc(freqBlockSize*freqBlockSize*3*sizeof(double));
            // for (i=0;i<freqBlockSize;i++){
            //     for(j=0;j<freqBlockSize;j++){
            //         blockimagefindex[(i*freqBlockSize+j)*3+0]=round((i*cos(angle))-(j*sin(angle)));
            //         blockimagefindex[(i*freqBlockSize+j)*3+1]=round((i*sin(angle))+(j*cos(angle)));
            //         blockimagefindex[(i*freqBlockSize+j)*3+2]=blockimagef[i*freqBlockSize+j];
            //     }
            // }

            // int indmax=blockimagefindex[0];
            // int indmin=blockimagefindex[1];
            // for(i=1;i<(freqBlockSize*freqBlockSize);i++){
            //     if(blockimagefindex[i*3+0]<indmin){
            //         indmin=blockimagefindex[i*3+0];
            //     }
            //     if(blockimagefindex[i*3+1]>indmax){
            //         indmax=blockimagefindex[i*3+1];
            //     }
            // }
            // if(indmax>freqBlockSize){
            //     for(i=0;i<(freqBlockSize*freqBlockSize);i++){
            //         blockimagefindex[i*3+1]=blockimagefindex[i*3+1]- indmax+freqBlockSize;
            //     }
            // }
            // if(indmin<0){
            //     for(i=0;i<(freqBlockSize*freqBlockSize);i++){
            //         blockimagefindex[i*3+0]=blockimagefindex[i*3+0]-indmin;
            //     }
            // }
            // for(i=0;i<(freqBlockSize*freqBlockSize);i++){
            //     blockimagef[i]=0.0;
            // }
            // printf("helol");
            // for(i=0;i<(freqBlockSize*freqBlockSize);i++){
            //     int new_i=blockimagefindex[i*3+0];
            //     int new_j=blockimagefindex[i*3+1];
            //     if(new_i>0 && new_i<freqBlockSize && new_j>0 && new_j<freqBlockSize){
            //         blockimagef[new_i*freqBlockSize+new_j]=blockimagefindex[i*3+2];
            //     }
            // }
            // int newsize = (int)round(freqBlockSize/sqrt(2.0));
            // double *croppedimagef = (double*)malloc(newsize*newsize*sizeof(double));
            // int offset = (int)round((freqBlockSize-newsize)/2);
            // for(i=0;i<newsize;i++){
            //     for(j=0;j<newsize;j++){
            //         croppedimagef[i*newsize+j]=blockimagef[(i+offset)*freqBlockSize+(j+offset)];
            //     }
            // }
            int newsize = freqBlockSize;

            double *proj=(double*)malloc(newsize*sizeof(double));
            for (i=0;i<newsize;i++){
                proj[i]=0.0;
            }
            for (i=0;i<newsize;i++){
                for (j=0;j<newsize;j++){
                    proj[j]+=blockimagef[i*newsize+j];
                }
            }
            double *proj2=(double*)malloc(newsize*sizeof(double));
            for (i=0;i<newsize;i++){
                double dilation;
                if((i<(freqWindowSize/2))){
                    dilation=proj[0];
                    for(j=1;j<=(i+(freqWindowSize/2));j++){
                        if (proj[j]>dilation){
                            dilation=proj[j];
                        }
                    }
                    dilation++;
                    proj2[i]=dilation;
                }
                else if(i>=(newsize-(freqWindowSize/2))){
                    dilation=proj[i-(freqWindowSize/2)];
                    for(j=(i-(freqWindowSize/2)+1);j<newsize;j++){
                        if (proj[j]>dilation){
                            dilation=proj[j];
                        }
                    }
                    dilation++;
                    proj2[i]=dilation;
                }
                else{
                    dilation=proj[i-(freqWindowSize/2)];
                    for(j=(i-(freqWindowSize/2)+1);j<=(i+(freqWindowSize/2));j++){
                        if(proj[j]>dilation){
                            dilation=proj[j];
                        } 
                    }
                    dilation++;
                    proj2[i]=dilation;
                }
            }
            double *temp=(double*)malloc(newsize*sizeof(double));
            for(i=0;i<newsize;i++){
                temp[i]=proj2[i]-proj[i];
                if (temp[i]<0){
                    temp[i]*=(-1);
                }
            }
            double projmean=0;
            for(i=0;i<newsize;i++){
                projmean+=proj[i];
            }
            projmean/=(double)newsize;
            double peak_thresh=2.0;
            bool *maxpts=(bool*)malloc(newsize*sizeof(bool));
            int colsmaxind=0;
            for(i=0;i<newsize;i++){
                if ((temp[i]<peak_thresh) && (proj[i]>projmean)){
                    colsmaxind++;
                    maxpts[i]=true;
                }
                else{
                    maxpts[i]=false;
                }
            }
            double *maxind=(double*)malloc(colsmaxind*sizeof(double));
            j=0;
            for(i=0;i<newsize;i++){
                if (maxpts[i]==true){
                    maxind[j]=i;
                    j++;
                }
            }

            if(colsmaxind<2){
                for (i=0;i<freqBlockSize;i++){
                    for(j=0;j<freqBlockSize;j++){
                        freqIm[(row+i)*width+(col+j)]=0;
                    }
                }
            }
            else{
                double wavelength = (maxind[colsmaxind-1]-maxind[0])/(colsmaxind-1);
                if ((wavelength>=(double)minWavelength) && (wavelength<=(double)maxWavelength)){
                    for (i=0;i<freqBlockSize;i++){
                        for(j=0;j<freqBlockSize;j++){
                            freqIm[(row+i)*width+(col+j)]=1/wavelength;
                        }
                    }
                }
                else{
                    for (i=0;i<freqBlockSize;i++){
                        for(j=0;j<freqBlockSize;j++){
                            freqIm[(row+i)*width+(col+j)]=0;
                        }
                    }
                }
            }
        }
    }
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if(mask[row*width+col]==false){
                freqIm[row*width+col]=0.0;
            } 
        }
    }
    double meanFrequency=0.0;
    int countfreq=0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if(freqIm[row*width+col]>0){
                meanFrequency+=freqIm[row*width+col];
                countfreq++;
            } 
        }
    }
    meanFrequency/=(double)(countfreq);
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if(mask[row*width+col]==false){
                freqIm[row*width+col]=0.0;
            } 
            else{
                freqIm[row*width+col]=meanFrequency;
            }
        }
    }
    return meanFrequency;
}

void createGaborFilter(double *gaborFilter, double *meshX, double *meshY, double *referenceFilter,double angleInclination, double meanFrequency,double sigmax, double sigmay, int gaborHalfSize, int gaborRows, int gaborCols){
    int angle, i, j;
    for(angle=0;angle<round(180/angleInclination);angle++){
        double ang=(((-1)*(angle*angleInclination+90))/180)*(PI);
        for(i=0;i<gaborRows;i++){
            for(j=0;j<gaborCols;j++){
                double xdash,ydash;
                meshX[i*gaborRows+j]=j-gaborHalfSize;
                meshY[i*gaborRows+j]=i-gaborHalfSize;
                xdash=(meshX[i*gaborRows+j]*cos(ang)) + (meshY[i*gaborRows+j]*sin(ang));
                ydash=((-1)*meshX[i*gaborRows+j]*sin(ang)) + (meshY[i*gaborRows+j]*cos(ang));
                referenceFilter[i*gaborRows+j]=exp(-(  (pow(xdash,2)/(sigmax*sigmax)) + (pow(ydash,2)/(sigmay*sigmay)) )) * cos(2*PI*meanFrequency*xdash);
                gaborFilter[angle*gaborCols*gaborRows + i*gaborCols + j] = referenceFilter[i*gaborRows+j];
            }
        }
    }
}

void roundOrientationValues(double *orientIm, double angleInclination,int width, int height){
    int row,col;
    int maxorient_index = round(180/angleInclination);
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            orientIm[row*width+col]= round(orientIm[row*width+col]/PI *180/angleInclination);
            if (((int)orientIm[row*width+col])<1){
                orientIm[row*width+col]+=maxorient_index;
            }
            if (((int)orientIm[row*width+col])>maxorient_index){
                orientIm[row*width+col]-=maxorient_index;
            }
        }
    }
}

void gaborIndexValueListInitiate(int *gaborIndexValueList, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            gaborIndexValueList[(row*width+col)*3+0]=row;
            gaborIndexValueList[(row*width+col)*3+1]=col;
            gaborIndexValueList[(row*width+col)*3+2]=255;
        }
    }
}

void gaborConvolution(double *freqIm, double *orientIm, double *normIm, double *gaborFilter, int *gaborIndexValueList, int gaborRows, int gaborCols, int gaborHalfSize,int width, int height){
    int row,col,i,j;
    for(row=gaborHalfSize;row<(height-gaborHalfSize);row++){
        for(col=gaborHalfSize;col<(width-gaborHalfSize);col++){

            if (freqIm[row*width+col]>0){
                int ori_ang=orientIm[row*width+col];
                int ang_i;
                
                for(ang_i=0;ang_i<60;ang_i+=19){
                    double gabor_sum=0.0;
                    for(i=-(gaborHalfSize);i<=gaborHalfSize;i++){
                        for(j=-(gaborHalfSize);j<=gaborHalfSize;j++){
                            gabor_sum+=normIm[(row+i)*width+(col+j)]*gaborFilter[ ((ang_i)*gaborRows*gaborCols) + ((gaborHalfSize+i)*gaborCols) + (gaborHalfSize+j)];
                        }
                    }
                    if(gabor_sum<-3){
                        gaborIndexValueList[(row*width+col)*3+0]=row;
                        gaborIndexValueList[(row*width+col)*3+1]=col;
                        gaborIndexValueList[(row*width+col)*3+2]=0;
                    }
                }
            }
            
        }
    }
}

void createFinalIm(int *finalIm, int *gaborIndexValueList, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            finalIm[(gaborIndexValueList[(row*width+col)*3+0])*width+(gaborIndexValueList[(row*width+col)*3+1])]=gaborIndexValueList[(row*width+col)*3+2];
        }
    }
}

void copyToOutput(PPMImage *image, int *finalIm, int width, int height){
    int i;
    for (i=0;i<width*height;i++){
        image->data[i].gray = (unsigned char)(finalIm[i]);
    }
}


//================================================================================================================================================================

int main(){
    int width,height,gradientSigma,blockSigma,orientSmoothSigma;
    int freqBlockSize, freqWindowSize, minWavelength, maxWavelength;
    int gaborHalfSize,gaborRows,gaborCols,kernelSize,gradientPaddedWidth,gradientPaddedHeight;
    double angleInclination, kx, ky, sigmax, sigmay;
    PPMImage *image;
    clock_t start,end;

    image = readPPM("Arch_8_O_v1.pgm");
    printf("%d %d", image->x,image->y);
    width = image->x;
    height = image->y;

    uint8_t *im = (uint8_t*)malloc(width*height*sizeof(uint8_t));
    
    imageCopy(im, image, width, height);
    
    start=clock();

    //Histogram Analysis
    //==========================================================================================
    histogramAnalysis(im,width,height);
    //==========================================================================================


    double *imageData = (double*)malloc(width*height*sizeof(double));
    bool *mask = (bool*)malloc(width*height*sizeof(bool));

    imageDouble(imageData,im,width,height);

    
    //Segmentation
    //==========================================================================================
    segmentation(imageData,mask,width,height);
    //==========================================================================================


    double *normIm = (double*)malloc(width*height*sizeof(double));


    //Normalization
    //==========================================================================================
    normalization(imageData,normIm,mask,width,height);
    //==========================================================================================


    //Orientation
    //==========================================================================================
    
    //local analysis (smaller kernel)
    gradientSigma=1;
    kernelSize=round(6*gradientSigma);
    if (kernelSize%2==0){
        kernelSize++;
    }
    gradientPaddedWidth = width + kernelSize-1;
    gradientPaddedHeight = height + kernelSize-1;

    double *GKernel=(double*)malloc(kernelSize*kernelSize*sizeof(double));   //Guassian Kernel
    double *gradientY=(double*)malloc(kernelSize*kernelSize*sizeof(double));        //gradient mtrx in y dir
    double *gradientX=(double*)malloc(kernelSize*kernelSize*sizeof(double));        //gradient mtrx in x dir
    double *Gx = (double*)malloc(width*height*sizeof(double));
    double *Gy = (double*)malloc(width*height*sizeof(double));
    double *paddedNormIm = (double*)malloc(gradientPaddedHeight*gradientPaddedWidth*sizeof(double));
    double *Gxx= (double*)malloc(width*height*sizeof(double));
    double *Gyy= (double*)malloc(width*height*sizeof(double));
    double *Gxy= (double*)malloc(width*height*(sizeof(double)));

    createGaussianKernelSigma(GKernel,gradientSigma,kernelSize);
    creategradientXgradientY(gradientX,gradientY,GKernel,kernelSize);    
    paddingZeroesOnEdges(paddedNormIm,normIm,kernelSize,gradientPaddedWidth,gradientPaddedHeight,width,height);
    convolve2d(paddedNormIm,gradientX,gradientY,Gx,Gy,kernelSize,gradientPaddedWidth,width,height);
    createGxxGyyGxy(Gxx,Gyy,Gxy,Gx,Gy,width,height);


    //Global analysis (bigger kernel)
    
    blockSigma = 7;
    kernelSize = round(6*blockSigma);
    if (kernelSize%2==0){
        kernelSize++;
    }
    gradientPaddedWidth=width+kernelSize-1;
    gradientPaddedHeight=height+kernelSize-1;

    double *GKernelBlock=(double*)malloc(kernelSize*kernelSize*sizeof(double));
    double *paddedGxx = (double*)malloc(gradientPaddedHeight*gradientPaddedWidth*sizeof(double));
    double *paddedGyy = (double*)malloc(gradientPaddedHeight*gradientPaddedWidth*sizeof(double));
    double *paddedGxy = (double*)malloc(gradientPaddedHeight*gradientPaddedWidth*sizeof(double));
    double *denom = (double*)malloc(height*width*sizeof(double));
    double *sin2theta = (double*)malloc(height*width*sizeof(double));
    double *cos2theta = (double*)malloc(height*width*sizeof(double));

    createGaussianKernelSigma(GKernelBlock,blockSigma,kernelSize);
    paddingReflectGradient(paddedGxx,Gxx,paddedGyy,Gyy,paddedGxy,Gxy,kernelSize,gradientPaddedWidth,gradientPaddedHeight,width,height);
    ndConvolutionGradient(paddedGxx,paddedGyy,paddedGxy,Gxx,Gyy,Gxy,GKernelBlock,gradientPaddedWidth,kernelSize,width,height);
    createSinCos(sin2theta,cos2theta,denom,Gxx,Gyy,Gxy,width,height);


    //creates orientation matrix 
    
    orientSmoothSigma = 7;
    kernelSize = round(6*orientSmoothSigma);
    if (kernelSize%2==0){
        kernelSize++;
    }
    gradientPaddedWidth=width+kernelSize-1;
    gradientPaddedHeight=height+kernelSize-1;

    double *GKernelOrient=(double*)malloc(kernelSize*kernelSize*sizeof(double));
    double *paddedsin2theta = (double*)malloc(gradientPaddedHeight*gradientPaddedWidth*sizeof(double));
    double *paddedcos2theta = (double*)malloc(gradientPaddedHeight*gradientPaddedWidth*sizeof(double));
    double *orientIm = (double*)malloc(height*width*sizeof(double));

    createGaussianKernelSigma(GKernelOrient,orientSmoothSigma,kernelSize);
    paddingReflectSinCos(paddedsin2theta,paddedcos2theta,sin2theta,cos2theta,kernelSize,gradientPaddedWidth,gradientPaddedHeight,width,height);
    ndConvolutionSinCos(paddedsin2theta,paddedcos2theta,GKernelOrient,sin2theta,cos2theta,gradientPaddedWidth,kernelSize,width,height);
    createOrientIm(orientIm,sin2theta,cos2theta,width,height);
    //==========================================================================================


    //Ridge Frequency
    //==========================================================================================

    freqBlockSize = 38;
    freqWindowSize = 5;
    minWavelength = 5;
    maxWavelength = 15;

    double *freqIm = (double*)malloc(height*width*sizeof(double));
    printf("hey");
    double meanFrequency = ridgeFrequency(normIm,orientIm,mask,freqIm,freqBlockSize,freqWindowSize,minWavelength,maxWavelength,width,height);
    printf("hi");
    //==========================================================================================


    //Gabor Filter (finally!)
    //==========================================================================================
    angleInclination = 3.0;
    kx=0.65;
    ky=0.65;
    meanFrequency=(double)(round(meanFrequency*100)/100);
    sigmax=1.0/meanFrequency*kx;
    sigmay=1.0/meanFrequency*ky;
    gaborHalfSize = round(3*sigmay);
    gaborRows=2*gaborHalfSize+1;
    gaborCols=2*gaborHalfSize+1;

    double *meshX=(double*)malloc(gaborRows*gaborCols*sizeof(double));
    double *meshY=(double*)malloc(gaborRows*gaborCols*sizeof(double));
    double *referenceFilter=(double*)malloc(gaborRows*gaborCols*sizeof(double));
    double *gaborFilter = (double *)malloc((180/angleInclination)*gaborRows*gaborCols*sizeof(double));
    int *gaborIndexValueList = (int*)malloc(width*height*3*sizeof(int));
    int *finalIm = (int*)malloc(height*width*sizeof(int));

    createGaborFilter(gaborFilter,meshX,meshY,referenceFilter,angleInclination,meanFrequency,sigmax,sigmay,gaborHalfSize,gaborRows,gaborCols);
    roundOrientationValues(orientIm,angleInclination,width,height);
    gaborIndexValueListInitiate(gaborIndexValueList,width,height);
    gaborConvolution(freqIm,orientIm,normIm,gaborFilter,gaborIndexValueList,gaborRows,gaborCols,gaborHalfSize,width,height);
    createFinalIm(finalIm,gaborIndexValueList,width,height);
    //==========================================================================================

    end=clock();

    double walltime = (end-start)/(double)CLOCKS_PER_SEC;
    printf("\nWalltime = %lf", walltime);

    copyToOutput(image,finalIm,width,height);

    writePPM("Arch_8_O_v1_gabor.pgm",image);
    printf("Press any key...");
    getchar();

    return 0;
}
