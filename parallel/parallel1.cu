#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<stdint.h>
#include<stdbool.h>
#include<cuda.h>

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

void copyimagetopointer(double *output, PPMImage *input, int width, int height){
    int i;
    for (i=0;i<(width*height);i++){
        output[i] = (double)input->data[i].gray;
    }
}

void initiliazing1dint(int *input, int length){
    int i;
    for(i=0;i<length;i++){
        input[i]=0;
    }
}

void histogramcreate(double *image, int *hist, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            hist[(int)image[row*width+col]]+=1;
        }
    }
}

int findingmin1dinthist(int *input, int histpercent, int length, int width, int height){
    int intmin, intsum=0;
    for(intmin=0;intmin<length;intmin++){
        intsum+=input[intmin];
        if ( ( (intsum*100) / (width*height) ) > histpercent ){
            break;
        }
    }
    return intmin;
}

int findingmax1dinthist(int *input, int histpercent, int length, int width, int height){
    int intmax, intsum=0;
    for(intmax=255;intmax>=0;intmax--){
        intsum+=input[intmax];
        if ( ( (intsum*100) / (width*height) ) > histpercent ){
            break;
        }
    }
    return intmax;
}

void histogramupdate(double *image, int intmax, int intmin, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){

            //assigning 0 pixel value to top x% black pixels
            if(image[row*width+col]<intmin){
                image[row*width+col]=0.0;
            }

            //assigning 255 pixel value to top x% white pixels
            else if (image[row*width+col]>intmax){
                image[row*width+col]=255.0;
            }

            //normalising the rest of the (100-2x)% pixels by raising the contrast
            else {
                image[row*width+col]=(double)(255.0*((int)image[row*width+col]-intmin)/(intmax-intmin)+0.5);
            }
        }
    }
}

double meanfunction(double *image, int width, int height){
    int row, col;
    double doublesum=0.0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            doublesum+=image[row*width+col];
        }
    }
    return doublesum;
}

double meanmaskfunction(double *image, bool *mask, int width, int height){
    int row, col;
    double doublesum=0.0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if (mask[row*width+col]==true){
                doublesum+=image[row*width+col];
            }
        }
    }
    return doublesum;
}

double stddevfunction(double *image, double doublemean, int width, int height){
    int row, col;
    double doublesum=0.0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            doublesum+=pow((image[row*width+col]-doublemean),2);
        }
    }
    return doublesum;
}

double stddevmaskfunction(double *image, bool *mask, double doublemean, int width, int height){
    int row, col;
    double doublesum=0.0;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if (mask[row*width+col]==true){
                doublesum+=pow((image[row*width+col]-doublemean),2);
            }
        }
    }
    return doublesum;
}

void normalise(double *image, double doublemean, double doublestddev, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            image[row*width+col] = (image[row*width+col] - doublemean)/ doublestddev;
        }
    }
}

void initializing2d(double *image, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            image[row*width+col] = 0.0;
        }
    }
}

//here height and width are for image to be copied and not the padded matrix
void copyimagetopleft(double *imagetocopy, double *paddedim, int paddedwidth, int paddedheight, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            paddedim[row*paddedwidth+col] = imagetocopy[row*width+col];
        }
    }
}

void localstddevsegmentation(double *paddedim, double *stddevim, int blocksize, int width, int height){
    int row, col, blockrow, blockcol;
    double doublesum, doublemean, doublestddev;

    for(row=0;row<height;row+=blocksize){
        for(col=0;col<width;col+=blocksize){

            //finding block mean
            doublesum = 0.0;
            for(blockrow=row;blockrow<row+blocksize;blockrow++){
                for(blockcol=col; blockcol<col+blocksize;blockcol++){
                    doublesum+=paddedim[blockrow*width+blockcol];
                }
            }
            doublemean=doublesum/(double)(blocksize*blocksize);

            // finding block standard deviation
            doublesum=0.0;
            for(blockrow=row;blockrow<row+blocksize;blockrow++){
                for(blockcol=col; blockcol<col+blocksize;blockcol++){
                    doublesum+=pow((paddedim[blockrow*width+blockcol]-doublemean),2);
                }
            }
            doublestddev=sqrt(doublesum/(double)(blocksize*blocksize));

            // assigning stddev values to corresponding block of std dev image/matrix
            for(blockrow=row;blockrow<row+blocksize;blockrow++){
                for(blockcol=col; blockcol<col+blocksize;blockcol++){
                    stddevim[blockrow*width+blockcol]=doublestddev;
                }
            }
        }
    }
}

int segmentation(double *stddevim, bool *mask, double threshold, int paddedwidth, int paddedheight, int width, int height){
    int row, col, totalrelevantpixels=0;

    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            if (stddevim[row*paddedwidth+col]>threshold){
                mask[row*width+col]=true;
                totalrelevantpixels++;
            }
            else{
                mask[row*width+col]=false;
            }
        }
    }

    return totalrelevantpixels;
}

void normalisesegmentation(double *image, double *normim, double doublemean, double doublestddev, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            normim[row*width+col] = (image[row*width+col] - doublemean)/ doublestddev;
        }
    }
}

double creatingGaussianKernel(double *GaussianKernel, int sigma, int gausskersize){
    double gsum, r, s=2.0*sigma*sigma;
    int x,y;

    for (x = -(gausskersize/2); x <= (gausskersize/2); x++) {
        for (y = -(gausskersize/2); y <= (gausskersize/2); y++) {
            r = sqrt(x * x + y * y);
            GaussianKernel[(x + (gausskersize/2))*gausskersize+(y + (gausskersize/2))] = (exp(-(r * r) / s)) / (PI * s);
            gsum += GaussianKernel[(x + (gausskersize/2))*gausskersize+(y + (gausskersize/2))];
        }
    }
    return gsum;
}

void normaliseGaussianKernel(double *GaussianKernel, double gsum, int gausskersize){
    int i,j;
    for (i = 0; i < gausskersize; i++){
        for (j = 0; j < gausskersize; j++){
            GaussianKernel[i*gausskersize+j] /= gsum;
        }
    }
}

void creatingGaussKerGradients(double *GaussianKernel, double *GaussKernelGradientX, double *GaussKernelGradientY, int gausskersize){
    int i,j;
    for (i=0;i<gausskersize;i++){
        for(j=0;j<gausskersize;j++){

            //finds gradient in y direction
            if (i==0){
                GaussKernelGradientY[i*gausskersize+j]=GaussianKernel[(i+1)*gausskersize+j]-GaussianKernel[i*gausskersize+j];
            }
            else if (i==(gausskersize-1)){
                GaussKernelGradientY[i*gausskersize+j]=GaussianKernel[i*gausskersize+j]-GaussianKernel[(i-1)*gausskersize+j];
            }
            else{
                GaussKernelGradientY[i*gausskersize+j]=(GaussianKernel[(i+1)*gausskersize+j]-GaussianKernel[(i-1)*gausskersize+j])/2.0;
            }

            //finds gradient in x direction
            if (j==0){
                GaussKernelGradientX[i*gausskersize+j]=GaussianKernel[i*gausskersize+(j+1)]-GaussianKernel[i*gausskersize+j];
            }
            else if (j==(gausskersize-1)){
                GaussKernelGradientX[i*gausskersize+j]=GaussianKernel[i*gausskersize+j]-GaussianKernel[i*gausskersize+(j-1)];
            }
            else{
                GaussKernelGradientX[i*gausskersize+j]=(GaussianKernel[i*gausskersize+(j+1)]-GaussianKernel[i*gausskersize+(j-1)])/2.0;
            }
        }
    }
}

void paddingzeroesonedges(double *imagetocopy, double *paddedim, int gausskersize, int paddedgausswidth, int paddedgaussheight, int width, int height){
    int row, col;
    for (row=0;row<paddedgaussheight;row++){
        for(col=0;col<paddedgausswidth;col++){
            if (row<(gausskersize/2) || col<(gausskersize/2) || col>=(paddedgausswidth-(gausskersize/2)) || row>=(paddedgaussheight-(gausskersize/2))){
                paddedim[row*paddedgausswidth+col]=0.0;
            }
            else{
                paddedim[row*paddedgausswidth+col]=imagetocopy[(row-(gausskersize/2))*width+(col-(gausskersize/2))];
            }
        }
    }
}

void convolutiongaussiankernelgradient(double *Gradientxx, double *Gradientyy, double *Gradientxy,double *paddedim, double *GaussKernelGradientx, double *GaussKernelGradienty, int gausskersize, int paddedgausswidth, int paddedgaussheight, int width, int height){
    int row, col, i, j;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            double sumx=0.0;
            double sumy=0.0;
            for (i=-(gausskersize/2);i<=(gausskersize/2);i++){
                for(j=-(gausskersize/2);j<=(gausskersize/2);j++){
                    sumx += paddedim[(row+i+(gausskersize/2))*paddedgausswidth+(col+j+(gausskersize/2))] * GaussKernelGradientx[(i+(gausskersize/2))*gausskersize+(j+(gausskersize/2))];
                    sumy += paddedim[(row+i+(gausskersize/2))*paddedgausswidth+(col+j+(gausskersize/2))] * GaussKernelGradienty[(i+(gausskersize/2))*gausskersize+(j+(gausskersize/2))];
                }
            }
            Gradientxx[row*width+col]=sumx*sumx;
            Gradientyy[row*width+col]=sumy*sumy;
            Gradientxy[row*width+col]=sumx*sumy;
        }
    }
}

void paddingreflectgradient(double *paddedGxx, double *paddedGyy, double *paddedGxy, double *Gxx, double *Gyy, double *Gxy, int gausskersize, int paddedwidth, int paddedheight, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            paddedGxx[(row+(gausskersize/2))*paddedwidth+(col+(gausskersize/2))]=Gxx[row*width+col];
            paddedGyy[(row+(gausskersize/2))*paddedwidth+(col+(gausskersize/2))]=Gyy[row*width+col];
            paddedGxy[(row+(gausskersize/2))*paddedwidth+(col+(gausskersize/2))]=Gxy[row*width+col];
        }
    }
    for(row=(gausskersize/2);row<(paddedheight-(gausskersize/2));row++){
        for(col=0;col<(gausskersize/2);col++){
            paddedGxx[row*paddedwidth+col]=paddedGxx[row*paddedwidth+(gausskersize-2-col)];
            paddedGyy[row*paddedwidth+col]=paddedGyy[row*paddedwidth+(gausskersize-2-col)];
            paddedGxy[row*paddedwidth+col]=paddedGxy[row*paddedwidth+(gausskersize-2-col)];
        }
        for(col=(paddedwidth-1);col>=(paddedwidth-(gausskersize/2));col--){
            paddedGxx[row*paddedwidth+col]=paddedGxx[row*paddedwidth+((paddedwidth-gausskersize)+(paddedwidth-col))];
            paddedGyy[row*paddedwidth+col]=paddedGyy[row*paddedwidth+((paddedwidth-gausskersize)+(paddedwidth-col))];
            paddedGxy[row*paddedwidth+col]=paddedGxy[row*paddedwidth+((paddedwidth-gausskersize)+(paddedwidth-col))];
        }
    }
    for(col=0;col<paddedwidth;col++){
        for(row=0;row<(gausskersize/2);row++){
            paddedGxx[row*paddedwidth+col]=paddedGxx[(gausskersize-2-row)*paddedwidth+col];
            paddedGyy[row*paddedwidth+col]=paddedGyy[(gausskersize-2-row)*paddedwidth+col];
            paddedGxy[row*paddedwidth+col]=paddedGxy[(gausskersize-2-row)*paddedwidth+col];
        }
        for(row=(paddedheight-1);row>=(paddedheight-(gausskersize/2));row--){
            paddedGxx[row*paddedwidth+col]=paddedGxx[((paddedheight-gausskersize)+(paddedheight-row))*paddedwidth+col];
            paddedGyy[row*paddedwidth+col]=paddedGyy[((paddedheight-gausskersize)+(paddedheight-row))*paddedwidth+col];
            paddedGxy[row*paddedwidth+col]=paddedGxy[((paddedheight-gausskersize)+(paddedheight-row))*paddedwidth+col];
        }
    }
}
__host__ __device__ double sqrt1( double a)
{
    return std::sqrt(a);
}

__host__ __device__ double pow1( double a, double b)
{
    return pow(a,b);
}

__global__ void ndconvolutionsincos(double *paddedGxx, double *paddedGyy, double *paddedGxy, double *sin2theta, double *cos2theta, double *GaussKernelBlockSigma, int gausskersize, int paddedwidth, int paddedheight, int width, int height){
    int row, col, i, j;
    row = blockIdx.y * blockDim.y + threadIdx.y;
	  col = blockIdx.x * blockDim.x + threadIdx.x;
    int sumxx=0.0;
    int sumyy=0.0;
    int sumxy=0.0;
    if(row<height && col<width)
    {
        for (i=-(gausskersize/2);i<=(gausskersize/2);i++){
            for(j=-(gausskersize/2);j<=(gausskersize/2);j++){
                sumxx+=paddedGxx[(row+i+(gausskersize/2))*paddedwidth+(col+j+(gausskersize/2))]*GaussKernelBlockSigma[(i+(gausskersize/2))*gausskersize+(j+(gausskersize/2))];
                sumyy+=paddedGyy[(row+i+(gausskersize/2))*paddedwidth+(col+j+(gausskersize/2))]*GaussKernelBlockSigma[(i+(gausskersize/2))*gausskersize+(j+(gausskersize/2))];
                sumxy+=paddedGxy[(row+i+(gausskersize/2))*paddedwidth+(col+j+(gausskersize/2))]*GaussKernelBlockSigma[(i+(gausskersize/2))*gausskersize+(j+(gausskersize/2))];
            }
        }
        sumxy = 2*sumxy;
        int denom = sqrt1(pow1(sumxy,2)+pow1((sumxx-sumyy),2)) + DBL_EPSILON;
        sin2theta[row*width+col] = sumxy/denom;
        cos2theta[row*width+col] = (sumxx-sumyy)/denom;
    }
}

void paddingreflectsincos(double *paddedsin2theta, double *paddedcos2theta, double *sin2theta, double *cos2theta, int gausskersize, int paddedwidth, int paddedheight, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            paddedsin2theta[(row+(gausskersize/2))*paddedwidth+(col+(gausskersize/2))]=sin2theta[row*width+col];
            paddedcos2theta[(row+(gausskersize/2))*paddedwidth+(col+(gausskersize/2))]=cos2theta[row*width+col];
        }
    }
    for(row=(gausskersize/2);row<(paddedheight-(gausskersize/2));row++){
        for(col=0;col<(gausskersize/2);col++){
            paddedsin2theta[row*paddedwidth+col]=paddedsin2theta[row*paddedwidth+(gausskersize-2-col)];
            paddedcos2theta[row*paddedwidth+col]=paddedcos2theta[row*paddedwidth+(gausskersize-2-col)];
        }
        for(col=(paddedwidth-1);col>=(paddedwidth-(gausskersize/2));col--){
            paddedsin2theta[row*paddedwidth+col]=paddedsin2theta[row*paddedwidth+((paddedwidth-gausskersize)+(paddedwidth-col))];
            paddedcos2theta[row*paddedwidth+col]=paddedcos2theta[row*paddedwidth+((paddedwidth-gausskersize)+(paddedwidth-col))];
        }
    }
    for(col=0;col<paddedwidth;col++){
        for(row=0;row<(gausskersize/2);row++){
            paddedsin2theta[row*paddedwidth+col]=paddedsin2theta[(gausskersize-2-row)*paddedwidth+col];
            paddedcos2theta[row*paddedwidth+col]=paddedcos2theta[(gausskersize-2-row)*paddedwidth+col];
        }
        for(row=(paddedheight-1);row>=(paddedheight-(gausskersize/2));row--){
            paddedsin2theta[row*paddedwidth+col]=paddedsin2theta[((paddedheight-gausskersize)+(paddedheight-row))*paddedwidth+col];
            paddedcos2theta[row*paddedwidth+col]=paddedcos2theta[((paddedheight-gausskersize)+(paddedheight-row))*paddedwidth+col];
        }
    }
}
__host__ __device__ double atan21( double a, double b)
{
    return atan2(a,b);
}

__global__ void ndconvolutionorient(double *paddedsin2theta, double *paddedcos2theta, double *orientim, double *GaussKernelBlockSigma, int gausskersize, int paddedwidth, int paddedheight, int width, int height){
    int row, col, i, j;
    row = blockIdx.y * blockDim.y + threadIdx.y;
	  col = blockIdx.x * blockDim.x + threadIdx.x;
      int sumsin,sumcos;
      if(row<height && col<width)
      {
        for (i=-(gausskersize/2);i<=(gausskersize/2);i++){
            for(j=-(gausskersize/2);j<=(gausskersize/2);j++){
            sumsin+=paddedsin2theta[(row+i+(gausskersize/2))*width+(col+j+(gausskersize/2))]*GaussKernelBlockSigma[(i+(gausskersize/2))*gausskersize+(j+(gausskersize/2))];
            sumcos+=paddedcos2theta[(row+i+(gausskersize/2))*width+(col+j+(gausskersize/2))]*GaussKernelBlockSigma[(i+(gausskersize/2))*gausskersize+(j+(gausskersize/2))];
            }
        }
        orientim[row*width+col] = (PI/2.0) + (atan21(sumsin,sumcos)/2.0);
      }
    
}

void ridgefrequency(double *normim, double *orientim, bool *mask, double *freqim, int blocksizefreq, double *blockimagefreq, double *orientblock, double *proj, double *proj2, bool *maxpts, int windowsizefreq, int minwavelength, int maxwavelength, int width, int height){
    int row, col, i, j;

    for(row=0;row<height-blocksizefreq;row+=blocksizefreq){
        for(col=0;col<width-blocksizefreq;col+=blocksizefreq){
            int colsmaxind;
            double projmean, dilation, wavelength, peak_thresh=2.0;

            for (i=0;i<blocksizefreq;i++){
                for(j=0;j<blocksizefreq;j++){
                    blockimagefreq[i*blocksizefreq+j]=normim[(row+i)*width+(col+j)];
                    orientblock[i*blocksizefreq+j]=orientim[(row+i)*width+(col+j)];
                }
            }

            for (i=0;i<blocksizefreq;i++){
                proj[i]=0.0;
            }

            for (i=0;i<blocksizefreq;i++){
                for (j=0;j<blocksizefreq;j++){
                    proj[j]+=blockimagefreq[i*blocksizefreq+j];
                }
            }

            for (i=0;i<blocksizefreq;i++){
                if((i<(windowsizefreq/2))){
                    dilation=proj[0];
                    for(j=1;j<=(i+(windowsizefreq/2));j++){
                        if (proj[j]>dilation){
                            dilation=proj[j];
                        }
                    } 
                }
                else if(i>=(blocksizefreq-(windowsizefreq/2))){
                    dilation=proj[i-(windowsizefreq/2)];
                    for(j=(i-(windowsizefreq/2)+1);j<blocksizefreq;j++){
                        if (proj[j]>dilation){
                            dilation=proj[j];
                        }
                    }
                }
                else{
                    dilation=proj[i-(windowsizefreq/2)];
                    for(j=(i-(windowsizefreq/2)+1);j<=(i+(windowsizefreq/2));j++){
                        if(proj[j]>dilation){
                            dilation=proj[j];
                        } 
                    }
                }
                dilation++;
                proj2[i]=dilation;
            }

            for(i=0;i<blocksizefreq;i++){
                proj2[i]=proj2[i]-proj[i];
                if (proj2[i]<0){
                    proj2[i]*=(-1);
                }
            }

            projmean=0.0;
            for(i=0;i<blocksizefreq;i++){
                projmean+=proj[i];
            }
            projmean/=(double)blocksizefreq;

            colsmaxind=0;
            for(i=0;i<blocksizefreq;i++){
                if ((proj2[i]<peak_thresh) && (proj[i]>projmean)){
                    colsmaxind++;
                    maxpts[i]=true;
                }
                else{
                    maxpts[i]=false;
                }
            }

            double *maxind=(double*)malloc(colsmaxind*sizeof(double));
            j=0;
            for(i=0;i<blocksizefreq;i++){
                if (maxpts[i]==true){
                    maxind[j]=i;
                    j++;
                }
            }

            if(colsmaxind>=2){
                wavelength = (maxind[colsmaxind-1]-maxind[0])/(colsmaxind-1);
                if ((wavelength>=(double)minwavelength) && (wavelength<=(double)maxwavelength)){
                    for (i=0;i<blocksizefreq;i++){
                        for(j=0;j<blocksizefreq;j++){
                            if(mask[(row+i)*width+(col+j)]==true){
                                freqim[(row+i)*width+(col+j)]=1/wavelength;
                            }
                        }
                    }
                }
            }
            free(maxind);
        }
    }
    
}

void gaborfiltercreate(double *gaborfilter, double sigmagabor, double meanfreq, int gaborsize, int gabfiltersize){
    int i,j,meshx,meshy;
    for(i=0;i<gabfiltersize;i++){
        for(j=0;j<gabfiltersize;j++){
            meshx=j-gaborsize;
            meshy=i-gaborsize;
            gaborfilter[i*gabfiltersize+j]=exp(-(  (pow(meshx,2)/(sigmagabor*sigmagabor)) + (pow(meshy,2)/(sigmagabor*sigmagabor)) )) * cos(2*PI*meanfreq*meshx);
        }
    }
}

void create3dgabor(double *gaborfilter3d, double *gaborfilter, double *rotatedgabor, double *rotind, double angleinc, int gabfiltersize){
    int i, j, angle, rot_i, rot_j;

    for(angle=0;angle<(round(180/angleinc));angle++){
        double ang=-(angle*angleinc+90);

        //stores new coordinates and values in another matrix
        for(i=0;i<gabfiltersize;i++){
            for(j=0;j<gabfiltersize;j++){
                rotind[(i*gabfiltersize+j)*3+0]=round(i*cos(ang)-j*sin(ang));
                rotind[(i*gabfiltersize+j)*3+1]=round(i*sin(ang)+j*cos(ang));
                rotind[(i*gabfiltersize+j)*3+2]=gaborfilter[i*gabfiltersize+j];
            }
        }

        //brings the rotated matrix to center
        for(i=0;i<(gabfiltersize*gabfiltersize);i++){
            rotind[i*3+0]=rotind[i*3+0]-(rotind[(int)round((double)(gabfiltersize*gabfiltersize)/2)*3+0]-(gabfiltersize/2));
            rotind[i*3+1]=rotind[i*3+1]-(rotind[(int)round((double)(gabfiltersize*gabfiltersize)/2)*3+1]-(gabfiltersize/2));
        }

        //intializes rotated matrix with 0s
        for(i=0;i<gabfiltersize;i++){
            for(j=0;j<gabfiltersize;j++){
                rotatedgabor[i*gabfiltersize+j]=0.0;
            }
        }

        //assigns rotated matrix pixel if its not out of bound keeping center same
        for(i=0;i<(gabfiltersize*gabfiltersize);i++){
            rot_i=rotind[i*3+0];
            rot_j=rotind[i*3+1];
            if(rot_i>=0 && rot_i<gabfiltersize && rot_j>=0 && rot_j<gabfiltersize){
                rotatedgabor[rot_i*gabfiltersize+rot_j]=rotind[i*3+2];
            }
        }

        for(i=0;i<gabfiltersize;i++){
            for(j=0;j<gabfiltersize;j++){
                gaborfilter3d[angle*gabfiltersize*gabfiltersize + i*gabfiltersize + j] = rotatedgabor[i*gabfiltersize+j];
            }
        }
    }
}

void roundorientation(double *orientim, double angleinc, int maxorientindex, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            //printf("\nangle=%lf",orientim[row*width+col]);
            orientim[row*width+col]= round((orientim[row*width+col]/PI) *(180/angleinc));
            //printf("%lf ", orientim[row*width+col]);
            if ((orientim[row*width+col])<1){
                orientim[row*width+col]+=maxorientindex;
            }
            if ((orientim[row*width+col])>maxorientindex){
                orientim[row*width+col]-=maxorientindex;
            }
            //printf("\nangle=%lf",orientim[row*width+col]);
        }
    }
}

void initializegaborind(int *gaborind, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            gaborind[(row*width+col)*3+0]=row;
            gaborind[(row*width+col)*3+1]=col;
            gaborind[(row*width+col)*3+2]=255;
        }
    }
}


__global__ void gaborconvolution(double *gaborfilter3d, double *freqim,double *normim, double *orientim, bool *mask, int *gaborind, int gabfiltersize, int gaborsize, int width, int height){
    int ang, i, j;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	  int col = blockIdx.x * blockDim.x + threadIdx.x;
    double gaborsum;
    //printf("\ngaborsize = %d",gaborsize);
    if(row>=gaborsize && row<(height-gaborsize) && col>=gaborsize && col<(width-gaborsize)){
        printf("hey");
        if (mask[row*width+col]==true){
            //int ori_ang=(int)orientim[row*width+col];
            for(ang=0;ang<60;ang+=10){
                printf("oi");
                gaborsum=0.0;
                for(i=-(gaborsize);i<=gaborsize;i++){
                    for(j=-(gaborsize);j<=gaborsize;j++){
                        gaborsum+=normim[(row+i)*width+(col+j)]*gaborfilter3d[ (ang*gabfiltersize*gabfiltersize) + ((gaborsize+i)*gabfiltersize) + (gaborsize+j)];
                    }
                }
                if(gaborsum<-3){
                    printf("success");
                    gaborind[(row*width+col)*3+0]=row;
                    gaborind[(row*width+col)*3+1]=col;
                    gaborind[(row*width+col)*3+2]=0;
                }
            }
        }
    }
}

void assigningoutput(double *finalim, int *gaborind, int width, int height){
    int row, col;
    for(row=0;row<height;row++){
        for(col=0;col<width;col++){
            finalim[(gaborind[(row*width+col)*3+0])*width+(gaborind[(row*width+col)*3+1])]=(double)gaborind[(row*width+col)*3+2];
        }
    }
}

void copyingoutput(double *finalim, PPMImage *image, int width, int height){
    int i;
    for (i=0;i<(width*height);i++){
        image->data[i].gray = (unsigned char)(finalim[i]);
    }
}


int main(){
    PPMImage *image;  //instance of greyscale image
    int width, height; //dimension of input image
    //int i, row, col;  //iterator variables
    int intmax, intmin; //integer max, min and sum variables
    int length,gaborsize,gabfiltersize,maxorientindex;
    int histpercent, totalrelevantpixels, gradientsigma, blocksigma, gausskersize;
    int paddedgausswidth, paddedgaussheight, blocksizefreq, windowsizefreq, minwavelength, maxwavelength;   
    double threshsegment, gausskersum, meanfreq, angleinc, kgabor, sigmagabor;
    int blocksizesegment, heightsegment, widthsegment;  // blocksize for segmentation, height and width of padded image for segmentation
    double doublesum, doublemean, doublestddev; //double sum, mean and std dev variables
    double *dev_paddedsin2theta, *dev_paddedcos2theta;
    double *dev_gaborfilter3d, *dev_finalim, *dev_freqim, *dev_normim, *dev_orientim;
    int *dev_gaborind;
    bool *dev_mask;
    double *dev_GaussianKernelBlockSigma;
    double *dev_paddedGradientxx,*dev_paddedGradientyy,*dev_paddedGradientxy;
    double *dev_sin2theta,*dev_cos2theta;

    
    cudaEvent_t start, end;
    //clock_t start, end;
    int Bcudaheight, Bcudawidth, Tcuda=32;
    double walltime;

    //storing imagedata in a ppmimage instance   or   reading image
    image = readPPM("Arch_1_O_v1.pgm");
    printf("%d %d", image->x,image->y);
    width = image->x;
    height = image->y;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    //start=clock();
    double *imagedata = (double*)malloc(width*height*sizeof(double));  //input image array
    double *normim = (double*)malloc(width*height*sizeof(double));     //normalised image array
    double *orientim = (double*)malloc(height*width*sizeof(double));   //orientation matrix
    double *freqim = (double*)malloc(height*width*sizeof(double));     //ridge frequency matrix
    double *finalim = (double*)malloc(height*width*sizeof(double));    //final image array

    //Cuda pointers declaration
    cudaMalloc((void**)&dev_finalim,width*height*sizeof(double));
    cudaMalloc((void**)&dev_freqim,width*height*sizeof(double));
    cudaMalloc((void**)&dev_normim,width*height*sizeof(double));
    cudaMalloc((void**)&dev_orientim,width*height*sizeof(double));
    cudaMalloc((void**)&dev_mask,width*height*sizeof(bool));

    //storing image data in an array pointer
    copyimagetopointer(imagedata,image,width,height);

    //histogram analysis
    //==============================================================================================
    int *hist = (int*)malloc(256*sizeof(int)); //histogram array

    length=256; //length of histogram array

    //initializing with 0s
    initiliazing1dint(hist,length);

    //creating histogram
    histogramcreate(imagedata, hist, width, height);

    //for finding top x% black pixels and top x% white pixels from the histogram analysis
    histpercent = 1; 

    //finding threshold greyscale pixel value for top x% black pixels
    intmin = findingmin1dinthist(hist,histpercent,length,width,height);

    //finding threshold greyscale pixel value for top x% white pixels
    intmax = findingmax1dinthist(hist,histpercent,length,width,height);

    //updating image values based on histogram analysis for increasing contrast
    histogramupdate(imagedata,intmax,intmin,width,height);

    //==============================================================================================
    
    
    //normalizing image
    //==============================================================================================

    //finding mean
    doublesum=meanfunction(imagedata, width, height);
    doublemean=doublesum/(double)(width*height);

    //finding standard deviation
    doublesum=stddevfunction(imagedata, doublemean, width, height);
    doublestddev=sqrt(doublesum/(double)(width*height));

    //normalising the image
    normalise(imagedata, doublemean, doublestddev, width, height);

    //==============================================================================================
    

    //segmentation
    //==============================================================================================

    //defining block size
    blocksizesegment = 3;

    //determining height and width of padded image such that it is a multiple of block size
    heightsegment = (int)(blocksizesegment*ceil((float)height/(float)blocksizesegment));
    widthsegment = (int)(blocksizesegment*ceil((float)width/(float)blocksizesegment));

    double *paddedimsegment = (double*)malloc(heightsegment*widthsegment*sizeof(double)); //padded image
    double *stddevim = (double*)malloc(heightsegment*widthsegment*sizeof(double)); //standard deviation matrix
    bool *mask = (bool*)malloc(width*height*sizeof(bool)); //mask matrix for relevant pixels

    //initializes with 0s
    initializing2d(paddedimsegment,widthsegment,heightsegment);
    initializing2d(stddevim,widthsegment,heightsegment);

    // im im im 0
    // im im im 0
    // im im im 0
    // 0  0  0  0
    // here image will be copied on the top left part of padded image
    // padding will be at bottom right
    copyimagetopleft(imagedata,paddedimsegment,widthsegment,heightsegment,width,height);

    //finds blockwise standard deviation
    localstddevsegmentation(paddedimsegment,stddevim,blocksizesegment,widthsegment,heightsegment);

    //threshold for standard deviation
    threshsegment=0.1;

    //assigns true value to only those pixels whose stddev>threshold
    //it returns total number of pixels where stddev>threshold or where mask == true
    totalrelevantpixels=segmentation(stddevim,mask,threshsegment,widthsegment,heightsegment,width,height);

    //finding global mean for just the segmented image or relevant pixels
    doublesum=meanmaskfunction(imagedata,mask,width,height);
    doublemean=doublesum/(double)(width*height);

    //finding global standard deviation for just the segmented image or relevant pixels
    doublesum=stddevmaskfunction(imagedata, mask, doublemean, width, height);
    doublestddev=sqrt(doublesum/(double)(width*height));

    //normalising image based on mean and stddev of relevant pixels or segmented image
    normalisesegmentation(imagedata, normim, doublemean, doublestddev, width, height);

    //==============================================================================================
    
    
    // Orientation
    //==============================================================================================
    
    //defining gradient sigma
    gradientsigma = 1;

    //defining size of gaussian kernel(its size is always odd)
    gausskersize=round((double)(6*gradientsigma))+1;

    double *GaussianKernel=(double*)malloc(gausskersize*gausskersize*sizeof(double));
    double *GaussKernelGradientY=(double*)malloc(gausskersize*gausskersize*sizeof(double));
    double *GaussKernelGradientX=(double*)malloc(gausskersize*gausskersize*sizeof(double));

    //creates gaussian kernel
    gausskersum=creatingGaussianKernel(GaussianKernel, gradientsigma, gausskersize);

    //normalises gaussian kernel
    normaliseGaussianKernel(GaussianKernel, gausskersum, gausskersize);

    //creating gradient gaussian kernels for x and y directions
    creatingGaussKerGradients(GaussianKernel,GaussKernelGradientX,GaussKernelGradientY,gausskersize);

    paddedgausswidth = width + gausskersize - 1;
    paddedgaussheight = height + gausskersize - 1;

    double *Gradientxx= (double*)malloc(width*height*sizeof(double));  //(gradient in x direction)^2
    double *Gradientyy= (double*)malloc(width*height*sizeof(double));  //(gradient in y direction)^2
    double *Gradientxy= (double*)malloc(width*height*(sizeof(double)));//(gradient in x direction) * (gradient in y direction)
    double *paddednormim = (double*)malloc(paddedgaussheight*paddedgausswidth*sizeof(double)); //padded image for convolution

    //padding 0s on edges
    // 0  0   0   0  0
    // 0 img img img 0
    // 0 img img img 0
    // 0 img img img 0
    // 0  0   0   0  0
    paddingzeroesonedges(normim,paddednormim,gausskersize,paddedgausswidth,paddedgaussheight,width,height);

    //convolution of padded image with 2 kernels simultaneously to obtain two outcomes (gx and gy)
    //and obtains matrix values for gx^2, gy^2 and gx*gy
    convolutiongaussiankernelgradient(Gradientxx,Gradientyy,Gradientxy,paddednormim,GaussKernelGradientX,GaussKernelGradientY,gausskersize,paddedgausswidth,paddedgaussheight,width,height);


    //block sigma 
    blocksigma=7;
    gausskersize=round((double)(6*blocksigma))+1;

    double *GaussianKernelBlockSigma=(double*)malloc(gausskersize*gausskersize*sizeof(double));

    gausskersum=creatingGaussianKernel(GaussianKernelBlockSigma, blocksigma, gausskersize);
    normaliseGaussianKernel(GaussianKernelBlockSigma, gausskersum, gausskersize);

    paddedgausswidth = width + gausskersize - 1;
    paddedgaussheight = height + gausskersize - 1;

    double *paddedGradientxx = (double*)malloc(paddedgaussheight*paddedgausswidth*sizeof(double));
    double *paddedGradientyy = (double*)malloc(paddedgaussheight*paddedgausswidth*sizeof(double));
    double *paddedGradientxy = (double*)malloc(paddedgaussheight*paddedgausswidth*sizeof(double));

    //padding is done such that boundary of matrix works as mirror
    // e d   d e f   f e
    // b a   a b c   c b
    //     ---------
    // b a | a b c | c b
    // e d | d e f | f e       Here, dotted lines are acting as mirrors
    // h g | g h i | i h       Everything outside of dotted line is padding
    //     ---------
    // h g   g h i   i h
    // e d   d e f   f e
    paddingreflectgradient(paddedGradientxx,paddedGradientyy,paddedGradientxy,Gradientxx,Gradientyy,Gradientxy,gausskersize,paddedgausswidth,paddedgaussheight,width,height);

    double *sin2theta = (double*)malloc(height*width*sizeof(double)); // Gxy  /  sqrt((4*(Gxy^2)) + ((Gxx-Gyy)^2))
    double *cos2theta = (double*)malloc(height*width*sizeof(double)); // (Gxx-Gyy)  /  sqrt((4*(Gxy^2)) + ((Gxx-Gyy)^2))
    
    cudaMalloc((void**)&dev_GaussianKernelBlockSigma,gausskersize*gausskersize*sizeof(double));
    cudaMalloc((void**)&dev_paddedGradientxx,paddedgaussheight*paddedgausswidth*sizeof(double));
    cudaMalloc((void**)&dev_paddedGradientyy,paddedgaussheight*paddedgausswidth*sizeof(double));
    cudaMalloc((void**)&dev_paddedGradientxy,paddedgaussheight*paddedgausswidth*sizeof(double));
    
    cudaMalloc((void**)&dev_sin2theta,height*width*sizeof(double));
    cudaMalloc((void**)&dev_cos2theta,height*width*sizeof(double));

    cudaMemcpy(dev_paddedGradientxx,paddedGradientxx,paddedgaussheight*paddedgausswidth*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_paddedGradientyy,paddedGradientyy,paddedgaussheight*paddedgausswidth*sizeof(double),cudaMemcpyHostToDevice);    
    cudaMemcpy(dev_paddedGradientxy,paddedGradientxy,paddedgaussheight*paddedgausswidth*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_GaussianKernelBlockSigma,GaussianKernelBlockSigma,gausskersize*gausskersize*sizeof(double),cudaMemcpyHostToDevice);

    
    Bcudaheight=ceil(height/Tcuda);
    Bcudawidth=ceil(width/Tcuda);
    dim3 blockspergridsincos(Bcudawidth,Bcudaheight);
    dim3 threadsperblocksincos(Tcuda,Tcuda);

    printf("\nHeldafefewgrsgrsgrwgrwgrwgrwooooooooo");
    //convolution of Gxx, Gyy and Gxy with bigger gaussian kernel for global analysis
    //the output is calculated and stored in sin2theta and cos2theta
    ndconvolutionsincos<<<blockspergridsincos,threadsperblocksincos>>>(paddedGradientxx,paddedGradientyy,paddedGradientxy,sin2theta,cos2theta,GaussianKernelBlockSigma,gausskersize,paddedgausswidth,paddedgaussheight,width,height);

    cudaMemcpy(dev_sin2theta,sin2theta,height*width*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(dev_cos2theta,cos2theta,height*width*sizeof(double),cudaMemcpyDeviceToHost);

    printf("\nHellooooooooo");

    //orient smooth sigma
    double *paddedsin2theta = (double*)malloc(paddedgaussheight*paddedgausswidth*sizeof(double));
    double *paddedcos2theta = (double*)malloc(paddedgaussheight*paddedgausswidth*sizeof(double));

    //padding is done such that boundary of matrix works as mirror
    // e d   d e f   f e
    // b a   a b c   c b
    //     ---------
    // b a | a b c | c b
    // e d | d e f | f e       Here, dotted lines are acting as mirrors
    // h g | g h i | i h       Everything outside of dotted line is padding
    //     ---------
    // h g   g h i   i h
    // e d   d e f   f e
    paddingreflectsincos(paddedsin2theta,paddedcos2theta,sin2theta,cos2theta,gausskersize,paddedgausswidth,paddedgaussheight,width,height);
    
    Bcudaheight=ceil(height/Tcuda);
    Bcudawidth=ceil(width/Tcuda);
    dim3 blockspergridorient(Bcudawidth,Bcudaheight);
    dim3 threadsperblockorient(Tcuda,Tcuda);

    cudaMalloc((void**)&dev_paddedsin2theta,paddedgaussheight*paddedgausswidth*sizeof(double));
    cudaMalloc((void**)&dev_paddedcos2theta,paddedgaussheight*paddedgausswidth*sizeof(double));
    
    cudaMemcpy(dev_paddedsin2theta,paddedsin2theta,paddedgaussheight*paddedgausswidth*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_paddedcos2theta,paddedcos2theta,paddedgaussheight*paddedgausswidth*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_GaussianKernelBlockSigma,GaussianKernelBlockSigma,gausskersize*gausskersize*sizeof(double),cudaMemcpyHostToDevice);
    //convolves sin2theta and cos2theta values with the big gaussian kernel for global analysis of orientation
    //since the sigma value is to be kept same the gaussian kernel will be same
    ndconvolutionorient<<<blockspergridorient,threadsperblockorient>>>(dev_paddedsin2theta,dev_paddedcos2theta,dev_orientim,dev_GaussianKernelBlockSigma,gausskersize,paddedgausswidth,paddedgaussheight,width,height);

    cudaMemcpy(orientim,dev_orientim,height*width*sizeof(double),cudaMemcpyDeviceToHost);
    printf("\nHellooooooooo Orient");
    //==============================================================================================


    //Ridge Frequency
    //==============================================================================================
    blocksizefreq=38;
    windowsizefreq=5;
    minwavelength=5;
    maxwavelength=15;

    //initializing with 0s
    initializing2d(freqim,width,height);

    double *blockimagefreq=(double*)malloc(blocksizefreq*blocksizefreq*sizeof(double));
    double *orientblock=(double*)malloc(blocksizefreq*blocksizefreq*sizeof(double));
    double *proj=(double*)malloc(blocksizefreq*sizeof(double));
    double *proj2=(double*)malloc(blocksizefreq*sizeof(double));
    bool *maxpts=(bool*)malloc(blocksizefreq*sizeof(bool));

    //finds ridge frequency of entire blocks and assigns it to relevant pixels or segmented image within block
    ridgefrequency(normim,orientim,mask,freqim,blocksizefreq,blockimagefreq,orientblock,proj,proj2,maxpts,windowsizefreq,minwavelength,maxwavelength,width,height);

    meanfreq = meanmaskfunction(freqim,mask,width,height);
    meanfreq/=(double)totalrelevantpixels;
    
    //==============================================================================================

 
    //Gabor Filter 
    //==============================================================================================
    angleinc=3.0;
    kgabor=0.65;

    //initializing with 0s
    initializing2d(finalim,width,height);

    meanfreq=(double)(round(meanfreq*100)/100);
    sigmagabor=1.0/meanfreq*kgabor;
    gaborsize=round(3*sigmagabor);
    gabfiltersize=2*gaborsize+1;

    double *gaborfilter=(double*)malloc(gabfiltersize*gabfiltersize*sizeof(double));

    //generating gabor filter
    gaborfiltercreate(gaborfilter,sigmagabor,meanfreq,gaborsize,gabfiltersize);

    double *gaborfilter3d = (double *)malloc(((int)(180/angleinc))*gabfiltersize*gabfiltersize*sizeof(double));
    double *rotatedgabor=(double*)malloc(gabfiltersize*gabfiltersize*sizeof(double));
    double *rotind=(double*)malloc(gabfiltersize*gabfiltersize*3*sizeof(double));

    //creates 3d matrix with rotated gabor filters at angles 90 to 270 in intervals of 3
    create3dgabor(gaborfilter3d,gaborfilter,rotatedgabor,rotind,angleinc,gabfiltersize);
    
    maxorientindex=(int)(180/angleinc);

    roundorientation(orientim,angleinc,maxorientindex,width,height);

    int *gaborind = (int*)malloc(width*height*3*sizeof(int));

    

    //initializing with 255s
    initializegaborind(gaborind,width,height);

    
 
    
    Bcudaheight=ceil(height/Tcuda);
    Bcudawidth=ceil(width/Tcuda);
    dim3 blockspergrid(Bcudawidth,Bcudaheight);
    dim3 threadsperblock(Tcuda,Tcuda);


    //double *dev_gaborfilter3d, *dev_finalim, *dev_freqim, *dev_normim, *dev_orientim, *dev_mask, *dev_gaborind;
    cudaMalloc((void**)&dev_gaborfilter3d,((int)(180/angleinc))*gabfiltersize*gabfiltersize*sizeof(double));
    
    cudaMalloc((void**)&dev_gaborind,width*height*3*sizeof(int));

    cudaMemcpy(dev_gaborfilter3d,gaborfilter3d,((int)(180/angleinc))*gabfiltersize*gabfiltersize*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_freqim,freqim,width*height*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_normim,normim,width*height*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_orientim,orientim,width*height*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mask,mask,width*height*sizeof(bool),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gaborind,gaborind,width*height*3*sizeof(int),cudaMemcpyHostToDevice);

    //convolves normalised image with appropriate orientation of gabor filter
    gaborconvolution<<<blockspergrid, threadsperblock>>>(dev_gaborfilter3d,dev_freqim,dev_normim,dev_orientim,dev_mask,dev_gaborind,gabfiltersize,gaborsize,width,height);
    //cudaDeviceSynchronize();
    cudaMemcpy(gaborind,dev_gaborind,width*height*3*sizeof(int),cudaMemcpyDeviceToHost);
    printf("\nHellooooooooo gabor");
    cudaEventRecord(end);
    //cudaEventSynchronize(end);
    //==============================================================================================

    
    //assigning values from index matrix to output final image
    assigningoutput(finalim,gaborind,width,height);
    //end=clock();
    //walltime = (end-start)/(double)CLOCKS_PER_SEC;
    float millisec = 0;
    cudaEventElapsedTime(&millisec,start,end);  //calculates time and stores in millisec
    walltime = (double)millisec;
    printf("\nWalltime = %lf",walltime);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    copyingoutput(finalim,image,width,height);
    writePPM("Arch_1_O_v1_gabor.pgm",image);
    printf("Press any key...");
    //getchar();

    free(imagedata);
    free(normim);
    free(orientim);
    free(freqim);
    free(mask);
    free(finalim);
    free(hist);
    free(paddedimsegment);
    free(stddevim);
    free(GaussianKernel);
    free(GaussKernelGradientX);
    free(GaussKernelGradientY);
    free(Gradientxx);
    free(Gradientyy);
    free(Gradientxy);
    free(paddednormim);
    free(paddedGradientxx);
    free(paddedGradientyy);
    free(paddedGradientxy);
    free(sin2theta);
    free(cos2theta);
    free(paddedsin2theta);
    free(paddedcos2theta);
    free(gaborfilter);
    free(gaborfilter3d);
    free(rotatedgabor);
    free(rotind);
    free(gaborind);
    free(blockimagefreq);
    free(orientblock);
    free(proj);
    free(proj2);
    free(maxpts);
    free(GaussianKernelBlockSigma);
    cudaFree(dev_gaborfilter3d);
    cudaFree(dev_finalim);
    cudaFree(dev_freqim);
    cudaFree(dev_normim);
    cudaFree(dev_orientim);
    cudaFree(dev_mask);
    cudaFree(dev_gaborind);

    return 0;
}
